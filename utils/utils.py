'''Helper functions'''
## TODO: differentiate our helpers from generic like DotDict, storing?

import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from tqdm import trange
from utils.gradients import get_grads

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def move_points(f, pts, repulsion=False, max_iter=200):
    '''
    f: torch.nn.Module
    pts: tensor of shape [N,D]
    NOTE: modifies pts in-place
    '''
    pts = pts.detach()
    pts.requires_grad = True

    # loss_over_iters = {}
    pbar = range(max_iter)

    lr = 1e-2
    optimizer = torch.optim.Adam([pts], lr=lr)
    for i in pbar:
        optimizer.zero_grad()
        vals_sq = f(pts).square()
        loss_vals = vals_sq.sum() / len(pts)

        if repulsion:
            dist_matrix = torch.cdist(pts, pts, compute_mode='donot_use_mm_for_euclid_dist')
            dist_matrix = dist_matrix + 1e8*torch.eye(*dist_matrix.shape) ## invalidate main diagonal containing self-distances
            loss_repulsion = (1/dist_matrix).sum() / (len(pts)**2)
            loss = loss_vals + 1e-6*loss_repulsion
            if torch.isnan(loss): print(f'exited before nan in iteration {i}'); break
            # pbar.set_description(f"loss: {loss.item():.2e}, "
            #                         f"loss_vals: {loss_vals.item():.2e}, "
            #                         f"loss_repulsion: {loss_repulsion.item():.2e}, "
            #                     )
        else:
            loss = loss_vals
            # pbar.set_description(f"loss: {loss.item():.2e}")
        loss.backward()
        optimizer.step()

        # loss_over_iters[i] = loss.item()
        # if loss<1e-10: break ## good enough

    # if plot:
    #     plt.plot(loss_over_iters.keys(), loss_over_iters.values())
    #     plt.semilogy()
    #     plt.show()

def norm_sq(x):
    '''Square-norm of N vectors gather in tensor of shape [N,D]'''
    return x.square().sum(1)

def norm(x):
    '''Norm of vectors N gather in tensor of shape [N,D]'''
    return norm_sq(x).sqrt()

def normalize(dfdx):
    '''Normalize dfdx to have normals. Both input and output are shape [N,D]'''
    return dfdx/norm(dfdx)[:,None]

def grads_to_basis(dfdx, dfdO): ## TODO: this and lstsq are probably a key part of the library
    '''Compute basis for LSTSQ from the gradients
    dfdx: tensor wrt coordinates of shape [N,D]
    dfdO: tensor wrt (latent) parameters of shape [N,P]
    '''
    return torch.einsum('i,ip->ip',-1/norm(dfdx),dfdO)

def solve_lstsq(B, c, lmbd=2, verbose=False, method='lstsq'):
    '''Least-squares of form Bx=c
    B: basis, tensor of shape [N,P]
    c: rhs, tensor of shape [N] (target deformation projected onto normal)
    '''
    assert method in ["lstsq", "normalsolve"], "Available methods are lstsq and normalsolve"
    with torch.no_grad():
        ## Driver depending on condition number and device
        ## driver=gels  is fast, but sometimes instable for high condition numbers
        ## driver=gelsd is much more stable, but also much slower
        if verbose: t0 = time()

        if method=='lstsq':
            if lmbd:
                ## Tikhonov regularization
                B_tik = torch.vstack([B, lmbd*torch.eye(len(B.T), device=B.device)])
                c_tik = torch.hstack([c,    torch.zeros(len(B.T), device=B.device)])
                lstsq = torch.linalg.lstsq(B_tik, c_tik, driver='gels')
            else:
                lstsq = torch.linalg.lstsq(B, c, driver='gels')
            sol = lstsq.solution
        elif method=='normalsolve':
            '''
            Assemble the normal equation ourselves,
            add regularization onto diagonal
            and torch.linalg.solve the square system.
            Less accurate than lstsq, but less memory requirements.
            For large lmbd they converge to the same.
            '''
            BTB = B.T@B
            BTc = B.T@c
            BTB += lmbd*torch.eye(len(BTB))
            sol = torch.linalg.solve(BTB, BTc)

        if verbose:
            print(f"Solved lstsq in {(time()-t0):.4f} s")
            print(f"Largest sol entry: {sol.abs().max():.2e}")
            approx = B@sol
            mean_res = (approx - c).square().sum()/len(c)
            print(f"Mean residual: {mean_res:.2e}")

        return sol

def update_params(model, dO):
    '''
    Manually update the model parameters.
    model: torch.nn.Module to be updated
    dO: parameter update (premultiplied with the learning rate), tensor of correct shape [P]
    NOTE: modifies model.parameters in-place
    '''
    shapes = [param.shape for param in model.parameters()]
    with torch.no_grad():
        idx_stop = 0
        for param, shape in zip(model.parameters(), shapes):
            ## Pick the correct part in the vector and reshape
            idx_start = idx_stop
            idx_stop = idx_start + torch.numel(param)
            y_param = dO[idx_start:idx_stop].reshape(shape)
            ## Update the weights with a learning rate
            new_param = param + y_param
            param.copy_(new_param)

def get_mean_curvature(F, H):
    '''
    Mean-curvature in D-dimensions
    F: grad(f) gradients, shape [N,D]
    H: hess(f) Hessian, shape [N,D,D]

    https://u.math.biu.ac.il/~katzmik/goldman05.pdf
    For a shape implicitly defined by f<0:
    - div(F/|F|) = -(FHF^T - |F|^2 tr(H)) / 2*|F|^3
    In <=3D we can expand the formula, if we want to validate https://www.archives-ouvertes.fr/hal-01486547/document
    fx, fy, fz = F.T
    fxx, fxy, fxz, fyx, fyy, fyz, fzx, fzy, fzz = H.flatten(start_dim=1).T
    k = (fx*fx*(fyy+fzz) + fy*fy*(fxx+fzz) + fz*fz*(fxx+fyy) - 2*(fx*fy*fxy+fx*fz*fxz+fy*fz*fyz)) / (2*(fx*fx+fy*fy+fz*fz).pow(3/2))
    '''
    ## Quadratic form
    FHFT = torch.einsum('bi,bij,bj->b', F, H, F)
    ## Trace of Hessian
    trH = torch.einsum('bii->b', H)
    ## Norm of gradient
    N = F.square().sum(1).sqrt()
    ## Mean-curvature
    return -(FHFT - N.pow(2)*trH) / (2*N.pow(3))

def filter_bbox(pts, bbox):
    bbox = torch.tensor(bbox).T
    mask = torch.logical_and(bbox[0]<pts, pts<bbox[1]).all(1)
    return pts[mask]

def filter_based_on_normals(f, pts, target, th=.3):
    dfdx = get_grads(f, pts, wrt_params=False)
    n = normalize(dfdx)
    mask = -(n*target).sum(1)>th ## TODO: might need to divide by norm, but ok
    return mask

from dgl.geometry import farthest_point_sampler
def get_farthest_points(pts, N):
    idxs = farthest_point_sampler(pts.unsqueeze(0), N)[0]
    return idxs

def get_sampled_points(f, N=10000, bbox=[[-.5,.5]]*3, th=.2, sigma=.01, N_new_rand=1000, optimize=False, verbose=False, max_iter=100):
    sampled_pts = torch.zeros(0,3)
    bbox = torch.tensor(bbox).T

    with torch.no_grad():
        for i in range(max_iter):
            if len(sampled_pts)>10:
                mask = get_farthest_points(sampled_pts, len(sampled_pts)//10)
                seeds = sampled_pts[mask]
            else: seeds = sampled_pts
            new_pts_pert = seeds + sigma*torch.randn(seeds.shape)
            new_pts_rand = torch.rand(N_new_rand,3)*(bbox[1]-bbox[0])+bbox[0]
            new_pts = torch.vstack([new_pts_pert, new_pts_rand])
            vals = f(new_pts).squeeze()
            mask = vals.abs()<th
            sampled_pts = torch.vstack([sampled_pts, new_pts[mask]])
            if verbose: print(len(sampled_pts), end=' ')
            if len(sampled_pts)>=N: break

    if optimize and len(sampled_pts):
        sampled_pts.requires_grad = True
        pbar = trange(100)
        optimizer = torch.optim.Adam([sampled_pts], lr=1e-3)
        for i in pbar:
            optimizer.zero_grad()
            vals_abs = f(sampled_pts).abs()
            mask = vals_abs>1e-3
            vals_abs = vals_abs[mask]
            loss = vals_abs.sum() / len(vals_abs)
            pbar.set_description(f"loss: {loss.item():.2e}, updated {len(vals_abs)} points")
            loss.backward()
            optimizer.step()
    return sampled_pts.detach()

if __name__=='__main__':
    from models.NN import GeneralNetBunny
    model = GeneralNetBunny(act='sin')
    pts = torch.rand(100, 2)
    move_points(model, pts)

    dfdO = torch.rand(100, 123)
    dfdx = torch.rand(100, 3)
    dn_target = torch.rand(100)
    B = grads_to_basis(dfdx, dfdO)
    solve_lstsq(B, dn_target, lmbd=0.1, verbose=True)
