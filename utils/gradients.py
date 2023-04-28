'''Make gradients into a module.'''

import torch

try:
    from functorch import make_functional, vmap, grad, hessian
    def get_grad_wrt_params(model, inputs):
        '''Compute gradients of batch using functorch.'''
        func, params = make_functional(model)

        def f(params, pt):
            return func(params, pt[None,:]).squeeze()

        vf = vmap(grad(f), in_dims=(None, 0))

        batched_grads = vf(params, inputs)
        batched_grads = [g.flatten(start_dim=1) for g in batched_grads] ## flatten batched grads per parameter

        return torch.hstack(batched_grads)
    
    def get_hessian(model, inputs):
        Hf = hessian(model)
        Hf_vmap = vmap(Hf)
        return Hf_vmap(inputs).squeeze(1)

except ImportError:
    print('functorch not available, falling back to looped pytorch')
    from torch.autograd import grad
    from torch.autograd.functional import hessian
    def get_grad_wrt_params(model, inputs):
        '''Compute gradients of batch using a pytorch loop.'''
        params = list(model.parameters())
        batched_grads = [grad(model(i), params) for i in inputs[:,None,:]]
        batched_grads = [torch.stack(shards).flatten(start_dim=1) for shards in zip(*batched_grads)]
        return torch.hstack(batched_grads)
    
    def get_hessian(model, inputs):
        return torch.stack([hessian(model, i) for i in inputs])

def get_grads(model, inputs, wrt_params=True, return_vals=False):
    '''
    Compute gradients wrt coordinates, latents, parameters.
    model:  torch.nn.Module so the footprint exposes parameters, forward.
            model.forward is expected to receive [B,D(+L)] shaped input as a single argument and return [B,1] output
    inputs: input coordinates and optionally latent codes, shape [B,D(+L)]
    wrt_params: whether to return gradients wrt NN parameters
    '''
    ## Prepare inputs
    inputs = inputs.detach()
    inputs.requires_grad = True
    # with torch.enable_grad
    ## Forward pass
    vals = model(inputs)
    ## Backward pass: TODO I think we can track grads of this if we add create_graph, retain_graph=True
    torch.autograd.backward(vals, torch.ones(len(vals),1, device=inputs.device))
    if not return_vals:
        if not wrt_params: return inputs.grad
        ## Gradient wrt parameters
        return inputs.grad, get_grad_wrt_params(model, inputs)
    else:
        if not wrt_params: return inputs.grad, vals
        ## Gradient wrt parameters
        return inputs.grad, get_grad_wrt_params(model, inputs), vals


if __name__=='__main__':
    

    ### NN.Net ###
    from models.NN import GeneralNetBunny
    model = GeneralNetBunny(act='sin')
    pts = torch.rand(100, 2)
    print(get_grad_wrt_params(model, pts).shape)
    # dfdx, dfdO = get_grads(model, pts)
    # print(dfdx.shape, dfdO.shape)
    # H = get_hessian(model, pts)
    # print(H.shape)


    # ### IM_NET ### TODO: this should be conciser maybe?
    # from IM_NET.modelAE import IM_AE
    # import h5py
    # from utils import DotDict

    # ## Most of these are not actually used but we need them to initialize the model
    # FLAGS = DotDict()
    # FLAGS.sample_dir = "samples/im_ae_out"
    # FLAGS.sample_vox_size = 64
    # FLAGS.dataset = "all_vox256_img"
    # FLAGS.checkpoint_dir = "checkpoint"
    # FLAGS.device = "cuda" ## optional

    # ## Load model
    # im_ae = IM_AE(FLAGS)
    # model_dir = "IM_NET/checkpoint/all_vox256_img_ae_64/IM_AE.model64-399.pth"
    # im_ae.im_network.load_state_dict(torch.load(model_dir, map_location=im_ae.device))
    # im_ae.im_network.eval()
    # model = im_ae.im_network.generator

    # ## Load latent codes
    # filename = "IM_NET/checkpoint/all_vox256_img_ae_64/all_vox256_img_train_z.hdf5"
    # with h5py.File(filename, "r") as f:
    #     batch_z = f['zs'][()]  # np.array
    # batch_z = torch.from_numpy(batch_z)
    # batch_z = batch_z.to(im_ae.device)

    # z = batch_z[0]
    # unit_from_grid = lambda x : (x-.5)/im_ae.real_size-.5
    # grid_from_unit = lambda y : (y+.5)*im_ae.real_size+.5

    # from time import time
    # t0 = time()
    # with torch.no_grad():
    #     out = im_ae.z2voxel(z)
    #     print(out.max())
    # print(f'{(time()-t0):.2f}')
    # import numpy as np
    # ## TODO: does it have to be so crooked? cant we just +-.5?
    # xmin = unit_from_grid(0)
    # xmax = unit_from_grid(im_ae.real_size)
    # ls = np.linspace(xmin,xmax,im_ae.real_size+2)
    # X_u, Y_u, Z_u = np.meshgrid(ls, ls, ls)
    # pts = np.vstack([X_u.ravel(), Y_u.ravel(), Z_u.ravel()]).T
    # pts = torch.from_numpy(pts).float().to(device=im_ae.device)
    # # pointz = torch.hstack([pts, z[None,:].expand(len(pts),len(z))])
    # # model(pointz)
    # # out = model.forward_chunk(pointz)
    # t0 = time()
    # with torch.no_grad():
    #     out = model.forward_z(pts, z)
    #     print(out.max())
    # print(f'{(time()-t0):.2f}')
    
    # print()
    # # pts = torch.rand(100, 3).to(im_ae.device)
    # # pointz = torch.hstack([pts, z[None,:].repeat(len(pts),1)])
    # # dfdx, dfdO = get_grads(model, pointz)
    # # print(dfdx.shape, dfdO.shape)