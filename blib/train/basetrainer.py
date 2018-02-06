import torch
from ..util import barit

class BaseTrainer(object):
    """Generic Base trainer object to inherit functionality from."""
    def __init__(self):
        self.training = True
        self.use_gpu = True
        self._models = {}
        self._losses = {}
    
    def cuda(self):
        """Sets the trainer to GPU mode. 
        
        If flagged, the data is cast to GPU before every iteration after being
        retrieved from the loader.
        """
        self.use_gpu = True

    def cpu(self):
        """Sets the trainer to CPU mode"""
        if torch.cuda.is_available():
            print('Setting to CPU mode even though GPU is available.')
        self.use_gpu = False

    def train(self):
        """Sets the trainer and models to training mode"""
        self.training = True
        for _, m in self._models.items():
            m.train()

    def eval(self):
        """Sets the trainer and models to inference mode"""
        self.training = False
        for _, m in self._models.items():
            m.eval()

    def _put_on_gpu(self, data):
        if self.use_gpu:
            if any([isinstance(data, x) for x in [set, list, tuple]]):
                data = type(data)(self._put_on_gpu(x) for x in data)
            else:
                data = data.cuda()
        return data

    def iteration(self, data):
        raise NotImplementedError

    def iterate(self, loader):
        """Performs an epoch of training or validation.
        
        Args:
            loader (iterable): The data loader.
        """
        for data in barit(loader, start='Training' if self.training else 'Validation'):
            data = self._put_on_gpu(data)
            yield data, self.iteration(data)
        return

    __call__ = iterate