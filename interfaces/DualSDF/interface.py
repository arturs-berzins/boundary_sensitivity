import torch
import importlib
import sys
import argparse
import yaml
import numpy as np
from interfaces.interface_base import InterfaceBase

## Since this module is expected to be called from the root folder,
## add this dir to the path as to not change anything in the files
import os
sys.path.insert(0, os.path.join(sys.path[0], 'interfaces/DualSDF'))



class Interface(InterfaceBase):
    def __init__(self, interface_args, device):
        super(Interface, self).__init__(interface_args, device)
        ## Define params and model if you want to use torch.func to compute the gradients
        # self.params = dict(self.trainer.deepsdf_net.named_parameters())
        # self.model = self.trainer.deepsdf_net
    

    def get_args(self):
        '''Parse the forwarded arguments in self.interface_args.'''
        # command line args
        parser = argparse.ArgumentParser(description='DualSDF interface.')
        parser.add_argument('config', type=str, help='The configuration file.')
        parser.add_argument('--pretrained', default=None, type=str, help='pretrained model checkpoint')
        
        args = parser.parse_args(self.interface_args.split()) ## parse arguments from the string https://stackoverflow.com/a/44482453
        
        def dict2namespace(config):
            namespace = argparse.Namespace()
            for key, value in config.items():
                if isinstance(value, dict):
                    new_value = dict2namespace(value)
                else:
                    new_value = value
                setattr(namespace, key, new_value)
            return namespace

        # parse config file
        with open(args.config, 'r') as f:
            config = yaml.load(f)
        config = dict2namespace(config)

        return args, config


    def _load(self):
        '''
        Set up and load the model.
        '''
        args, cfg = self.get_args()

        trainer_lib = importlib.import_module(cfg.trainer.type)
        self.trainer = trainer_lib.Trainer(cfg, args, self.device)
    
        if args.pretrained is not None:
            self.trainer.resume_demo(args.pretrained)
        else:
            self.trainer.resume_demo(cfg.resume.dir)
        
        self.trainer.deepsdf_net.eval()
    

    def f(self, pointz):
        '''
        A function to evaluate the generative model with specific input and output signatures.
        Args:
            pointz: torch.Tensor of shape (N, D+L) where 
                N: number of samples
                D: spatial dimensions (likely 3)
                L: latent dimensions
        Returns:
            torch.Tensor of shape (N, 1) of the implicit function evaluated at pointz
        '''
        inp = pointz.unsqueeze(0)
        return self.trainer.deepsdf_net(inp).squeeze(0)
        
    
    def get_f(self, z):
        '''
        Make a function to evaluate the generative model over spatial coordinates but with a fixed latent feature.
        This is useful for marching cubes.
        Args:
            z: torch.Tensor of shape (D) or (1, D)
        Returns:
            A function f which receives only spatial coordinates and returns the implicit function values: 
            Args:
                p: torch.Tensor of shape (N,D)
            Returns:
                torch.Tensor of shape (N,1)
        '''
        z = z.unsqueeze(0)
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
        def f(p):
            N = p.size(0)
            inp = torch.cat([z.expand(-1,N,-1), p.unsqueeze(0)], dim=-1)
            dists = self.trainer.deepsdf_net(inp) # [1 N 1]
            dists = dists.reshape(-1, 1)
            return dists
        return f


    def get_random_feature(self):
        '''
        Get a random feature of a shape. If not available, generate a random latent feature.
        Args:
            None
        Returns:
            torch.Tensor of shape (1,L) where L is the latent dimensions
        '''
        num_features = self.trainer.get_known_latent(None)
        feature = self.trainer.get_known_latent(np.random.choice(num_features)) # 1, 128
        return feature