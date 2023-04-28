import os
import torch
import argparse
import json
import importlib

import http.server
import socketserver
from functools import partial

from utils import utils
from utils.gradients import get_grads
from utils.utils_exp import get_mesh

import numpy as np

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, interface, mc_resolution, decimals, *args, **kwargs):
        self.interface = interface
        self.start_path = os.path.abspath('pyjs3d')
        self.mc_resolution = mc_resolution
        self.decimals = decimals
        super().__init__(*args, **kwargs)
        
    def send_head(self):
        path = self.translate_path(self.path)
        f = None
        try:
            if not path.startswith(self.start_path):
                raise IOError
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            #self.send_error(404, "File not found")
            self.send_response(301)
            self.send_header("Location", "/pyjs3d/html/webgl_gui.html")
            self.end_headers()
            return None
        ctype = self.guess_type(path)
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f
        
    def do_POST(self):
        length = int(self.headers.get_all('content-length')[0])
        data_string = self.rfile.read(length)
        data = json.loads(data_string)

        if self.path == '/deform':
            '''This is where boundary sensitivity happens'''
            handles = torch.tensor(data['locations'], dtype=torch.float32, device=self.interface.device)
            dn_target = torch.tensor(data['displacements'], dtype=torch.float32, device=self.interface.device) ## TODO: the dn that we get might not be along the same normal
            L = torch.tensor(data['feature'], dtype=torch.float32, device=self.interface.device).squeeze()
            
            ## Flip the handles
            handles[:,1] *= -1

            ## Compute dfdx and dfdL at points
            pointz = torch.hstack([L[None,:].repeat(len(handles),1), handles])
            dfdLx = get_grads(self.interface.f, pointz, wrt_params=False)
            dfdL, dfdx = torch.split(dfdLx, [len(dfdLx.T)-3,3], dim=1)

            ## LSTSQ
            B = utils.grads_to_basis(dfdx, dfdL)
            dL = utils.solve_lstsq(B, dn_target, lmbd=.1, verbose=False)
            # dn_approx = B@dL

            feature = L + dL
            verts, faces, normals, feature = self.get_new_shape(feature=feature)
            self.send_shape(verts, faces, normals, feature)

        elif self.path == '/get_mesh': ## this should be GET, but it doesn't really matter
            verts, faces, normals, feature = self.get_new_shape()
            self.send_shape(verts, faces, normals, feature)
        else:
            print('Unknow POST path')
    
    def get_new_shape(self, feature=None):
        if feature is None:
            feature = self.interface.get_random_feature()

        f = self.interface.get_f(feature)
        verts, faces, normals = get_mesh(f, N=self.mc_resolution, device=self.interface.device, flip_faces=0, return_normals=1) ## TODO: N into args
        ## Flip y axis
        verts[:,1] *= -1
        normals[:,1] *= -1
        faces = faces[:,::-1]
        return verts, faces, normals, feature

    def send_shape(self, verts, faces, normals, feature):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.flush_headers()
        ## Reduce the precision so we dont have to send that much over the connection
        ## This is mostly relevant when hosting
        self.wfile.write(json.dumps({
            'verts': np.round(verts, self.decimals).tolist(),
            'faces': faces.tolist(),
            'normals': np.round(normals, self.decimals).tolist(),
            'feature': feature.tolist(),
            }).encode())


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

def run(server_class=http.server.HTTPServer, handler_class=http.server.BaseHTTPRequestHandler, interface=None, mc_resolution=64, decimals=8, port=1234):
    handler = partial(handler_class, interface, mc_resolution, decimals)
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler)
    httpd.trainer = interface
    httpd.serve_forever()

def get_args():
    parser = argparse.ArgumentParser(description='GUI demo of semantic editing with boundary sensitivity.')
    parser.add_argument('interface', type=str, help='Path to the interface root folder.')
    parser.add_argument('--interface_args', default=None, type=str, help='Argument string forwarded to the model interface.')
    parser.add_argument('--mc_resolution', default=64, type=int, help='Resolution of marching cubes for detail/compute trade-off.')
    parser.add_argument('--decimals', default=6, type=int, help='Number of decimal places to round vertices and normals before sending them over the connection. Less means smaller JSONs.')
    parser.add_argument('--device', default=None, type=str, help='CPU or CUDA. If unspecified, CUDA will be used if available.')
    parser.add_argument('--port', default=1234, type=int)
    args = parser.parse_args()
    return args

   
if __name__ == "__main__":
    args = get_args()
    if args.device is None: args.device = 'cuda' if torch.cuda.is_available else 'cpu'
    device = torch.device(args.device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    ## Change the working directory to that of the interface, so loading and imports work 
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, args.interface))

    ## Load the module from file path https://stackoverflow.com/a/46198651
    spec = importlib.util.spec_from_file_location('interface', 'interface.py')
    interface_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface_lib)
    ## Alternatively, use module notation, but then we have first have to convert (os-specific) path to namespace
    # interface_lib = importlib.import_module('interfaces.DualSDF.interface')
    interface = interface_lib.Interface(args.interface_args, device)
    ## Change back to this directory
    os.chdir(cwd)
    
    run(http.server.HTTPServer, TestHandler, interface, args.mc_resolution, port=args.port)
