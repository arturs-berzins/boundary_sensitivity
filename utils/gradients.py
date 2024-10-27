'''Function for computing the gradients.'''

import torch


def get_grads(model, inputs):
    '''
    Compute gradients wrt coordinates, latents, parameters.
    model:  torch.nn.Module so the footprint exposes parameters, forward.
            model.forward is expected to receive [B,D(+L)] shaped input as a single argument and return [B,1] output
    inputs: input coordinates and optionally latent codes, shape [B,D(+L)]
    '''
    ## Prepare inputs
    inputs = inputs.detach()
    inputs.requires_grad = True
    ## Forward pass
    vals = model(inputs)
    ## Backward pass
    torch.autograd.backward(vals, torch.ones(len(vals),1, device=inputs.device))
    return inputs.grad


## If you want to use torch.func instead, this is how it can be done.
## The interface should define model and params. E.g. see __init__ in DualSDF.interface
# from torch.func import functional_call, jacrev, vmap
# def get_grad_func(interface, inputs):
#     def f(params, xL):
#         return functional_call(interface.model, params, xL)
#     f_xL = jacrev(f, argnums=1)  ## params, [nxz] -> [nxz, ny]
#     vf_xL = vmap(f_xL, in_dims=(None, 0), out_dims=(0))  ## params, [bxz, nxz] -> [bxz, ny, nxz]
#     dfdLx = vf_xL(interface.params, inputs).squeeze(1)
#     return dfdLx