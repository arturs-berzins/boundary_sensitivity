import abc

class InterfaceBase():
    def __init__(self, interface_args, device):
        self.interface_args = interface_args
        self.device = device
        self._load()
    

    @abc.abstractmethod
    def _load(self):
        '''
        Set up and load the model.
        '''
        pass
    

    @abc.abstractmethod
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
        pass
    

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
        def f(p):
            pass
        return f


    @abc.abstractmethod
    def get_random_feature(self):
        '''
        Get a random feature of a shape. If not available, generate a random latent feature.
        Args:
            None
        Returns:
            torch.Tensor of shape (1,L) where L is the latent dimensions
        '''
        pass