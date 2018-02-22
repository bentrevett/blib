import torch
from torch.autograd import Variable
from tqdm import tqdm
from warnings import warn
from collections import defaultdict

class Trainer:
    """
    Generic trainer which everything else should take from.
    """
    def __init__(self,
                 dataloaders,
                 model,
                 optimizer,
                 criterion,
                 metrics = None,
                 scheduler = None,
                 clip = None,
                 n_inp = 1,
                 n_out = 1,
                 use_gpu = torch.cuda.is_available()):
        """
        - Dataloaders are the torch.utils.data.DataLoader's for train/val/test data. If only two dataloaders are supplied, val == test.

        - clip is value to clip gradients
        - n_inp is the number of inputs
        - n_out is the number of outputs
        """
        
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.clip = clip
        self.n_inp = n_inp
        self.n_out = n_out
        self.use_gpu = use_gpu

        self.losses = defaultdict(list) #holds all the losses

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.process_dataloaders(dataloaders)

    def process_dataloaders(self, dataloaders):
        """
        You may pass either 1, 2, or 3 dataloaders to the Trainer. 
        This handles all 3 cases.
        If you only pass 1, that same dataloader is used for train, val and test.
        If you only pass 2, the first is used as train, the second as  BOTH val and test.
        If you pass 3, they are used for train, val and test, respectively.

        ORDER MATTERS! The expected order is train, val, test!

        This also checks if the dataloader has the correct number of inputs/outputs.
        """
        
        assert type(dataloaders) is tuple, f'dataloaders must be a tuple of (train, val, test) dataloaders'
        assert len(dataloaders) < 4, f'Can only handle 3 dataloaders, you gave {len(dataloaders)}'

        for dl in dataloaders:
            for item in dl:
                assert len(item) == (self.n_inp + self.n_out), f'Incorrect number of inputs/outputs in the dataloader!'

        if len(dataloaders) == 1:
            warn('Only one dataloader supplied, setting test = val = train!')
            return dataloaders[0], dataloaders[0], dataloaders[0]

        elif len(dataloaders) == 2:
            warn('Only two dataloaders supplied, setting test = val!')
            return dataloaders[0], dataloaders[1], dataloaders[1]

        else:
            return dataloaders[0], dataloaders[1], dataloaders[2]

    def train(self):
        
        self.model.train()
        self.mode = 'Training'
        loss = self.iteration(self.train_dataloader)
        self.losses['train_loss'].append(loss)

    def validate(self):
        
        self.model.eval()
        self.mode = 'Validating'
        loss = self.iteration(self.val_dataloader)
        self.losses['val_loss'].append(loss)

    def test(self):
        
        self.model.eval()
        self.mode = 'Testing'
        loss = self.iteration(self.test_dataloader)
        self.losses['test_loss'].append(loss)

    def iteration(self, dataloader):

        for i, data in enumerate(tqdm(dataloader, desc=self.mode), start=1):
            
            X = data[:self.n_inp]
            y = data[self.n_inp:]

            X = [Variable(x, volatile = False if self.mode == 'Training' else True) for x in X]
            y = [Variable(_y, volatile = False if self.mode == 'Training' else True) for _y in y]
            
            if self.use_gpu:
                X = [x.cuda() for x in X]
                y = [_y.cuda() for _y in y] 

            if self.mode == 'Training':
                self.optimizer.zero_grad()

            y_pred = self.model(*X)

            if self.n_out > 1:
                assert len(y_pred) == len(y), f'Model output shape is {y_pred.shape}, target shape is {y.shape}'
                loss = sum([self.criterion(_y_pred, _y.squeeze()) for _y_pred, _y in zip(y_pred, y)])
            else:
                loss = self.criterion(y_pred, y[0].squeeze())

            if self.mode == 'Training':
                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
                self.optimizer.step()

        return loss.data[0]/len(dataloader)
        
    
        