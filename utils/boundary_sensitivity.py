'''Functions to perform boundary sensitivity.'''

import torch
from time import time


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