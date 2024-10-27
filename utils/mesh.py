'''
Utils for meshing the implicit surface.
'''

import torch
import numpy as np
from skimage import measure


def mc(V, level=0, return_normals=False):
    '''Marching cubes'''
    V = V.detach().cpu().numpy()
    verts, faces, normals, values = measure.marching_cubes(V, level)
    if return_normals: return verts, faces, -normals
    return verts, faces


def get_mesh(model, N=128, device='cpu', bbox_min=(-1,-1,-1), bbox_max=(1,1,1), chunks=1, flip_faces=False, level=0, return_normals=0):
    '''
    NOTE: for chunks>1 there are duplicate vertices at the seams. Merge them.
    '''
    with torch.no_grad():
        verts_all = []
        faces_all = []
        if return_normals: normals_all = []
        x0, y0, z0 = bbox_min
        x1, y1, z1 = bbox_max
        dx, dy, dz = (x1-x0)/chunks, (y1-y0)/chunks, (z1-z0)/chunks
        nV = 0 # nof vertices
        for i in range(chunks):
            for j in range(chunks):
                for k in range(chunks):
                    xmin, ymin, zmin = x0+dx*i, y0+dy*j, z0+dz*k
                    xmax, ymax, zmax = xmin+dx, ymin+dy, zmin+dz
                    lsx = torch.linspace(xmin, xmax, N//chunks, device=device)
                    lsy = torch.linspace(ymin, ymax, N//chunks, device=device)
                    lsz = torch.linspace(zmin, zmax, N//chunks, device=device)
                    X, Y, Z = torch.meshgrid(lsx, lsy, lsz, indexing='ij')
                    pts = torch.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                    vs = model(pts.to(device=device))
                    V = vs.reshape(X.shape).cpu()
                    res = mc(V, level=level, return_normals=return_normals)
                    if return_normals: verts, faces, normals = res
                    else: verts, faces = res
                    verts[:,0] = (xmax-xmin)*verts[:,0]/(V.shape[0]-1) + xmin
                    verts[:,1] = (ymax-ymin)*verts[:,1]/(V.shape[1]-1) + ymin
                    verts[:,2] = (zmax-zmin)*verts[:,2]/(V.shape[2]-1) + zmin
                    verts_all.append(verts)
                    faces_all.append(faces+nV)
                    if return_normals: normals_all.append(normals)
                    nV += len(verts)
    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)
    
    if return_normals: normals_all = np.vstack(normals_all)
    if flip_faces: faces = faces[:,::-1]
    if return_normals: return verts, faces, normals
    return verts, faces