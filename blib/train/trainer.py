import torch
from torch.autograd import Variable
from tqdm import tqdm
from warnings import warn
from collections import defaultdict
import os

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
                 use_gpu = torch.cuda.is_available(),
                 load_path = None,
                 save_path = None):
        """
        - Dataloaders are the torch.utils.data.DataLoader's for train/val/test data. If only two dataloaders are supplied, val == test.

        - clip is value to clip gradients
        - n_inp is the number of inputs
        - n_out is the number of outputs
        """
        if clip is not None:
            assert clip>0, 'clip should be greater than 0'
        assert n_inp > 0, 'Need at least one input'
        assert n_out > 0, 'Need at least one output'

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
        self.load_path = load_path
        self.save_path = save_path

        self.losses = defaultdict(list) #holds all the losses
        self.best_val_loss = float('inf')

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.process_dataloaders(dataloaders)

        if self.load_path is not None:
            assert os.path.isfile(load_path), 'Supplied weights path doesn\'t exist'
            self.model.load_state_dict(torch.load(self.load_path))

        if self.use_gpu:
            model = model.cuda()

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

    def run(self, n_epochs, metric='val_loss', patience=float('inf'), patience_mode='min', test=True, verbose=True):
        """
        Does the train, val, test loop

        n_epochs: the number of epochs to do train -> val loop for
        metric: the metric to use for early stopping
        patience: the patience value for early stopping
        patience_mode: 'min' if you want to metric to be decreasing (loss), 'max' if you want it to increase (acc/F1)
        test: if `run` should automatically run on the test data
        verbose: if the model should print the results for each train -> val loop
        """
        if patience_mode == 'min':
            best_run_metric = float('inf') #early stopping loss
        elif patience_mode == 'max':
            best_run_metric = 0

        patience_count = 0 #how many epochs have gone by without improvement in early stopping metric

        for i in range(n_epochs):
            self.train()
            self.validate()
            if verbose:
                print(f"Epoch: {i+1}, Train Loss: {self.losses['train_loss'][-1]:.3f}, Train Acc: {self.losses['train_acc'][-1]*100:.2f}%, Val. Loss: {self.losses['val_loss'][-1]:.3f}, Val Acc. {self.losses['val_acc'][-1]*100:.2f}%")
            if patience_mode == 'min':
                if self.losses[metric][-1] < best_run_metric:
                    best_run_metric = self.losses[metric][-1]
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count > patience:
                        print('Stopping early.')
                        break
            elif patience_mode == 'max':
                if self.losses[metric][-1] > best_run_metric: 
                    best_run_metric = self.losses[metric][-1]
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count > patience:
                        print('Stopping early.')
                        break


        if test:
            self.test()
            if verbose:
                print(f"Test Loss: {self.losses['test_loss'][-1]:.3f}, Test Acc.: {self.losses['test_acc'][-1]*100:.2f}%")

    def train(self):
        """
        Runs a single epoch while updating parameters
        """
        
        self.model.train() #turn back on do/bn
        self.mode = 'Training'
        loss, acc = self._iteration(self.train_dataloader) #run single pass
        self.losses['train_loss'].append(loss) #append losses
        self.losses['train_acc'].append(acc) #append accuracy

    def validate(self):
        """
        Runs a single epoch while not updating any parameters
        Updates scheduler
        Saves parameters if best validation loss so far
        """
        
        self.model.eval() #turn off do/bn
        self.mode = 'Validating'
        loss, acc = self._iteration(self.val_dataloader) #run single pass
        self.losses['val_loss'].append(loss) #append losses
        self.losses['val_acc'].append(acc) #append accuracy

        if self.scheduler is not None:
            self.scheduler.step(loss) #update LR

        if self.save_path is not None and loss < self.best_val_loss:
            torch.save(self.model.state_dict(), self.save_path) #save params if best loss
            self.best_val_loss = loss

    def val(self):
        """
        Alias for validate
        """
        self.validate()

    def eval(self):
        """
        Alias for validate
        """
        self.validate()

    def evaluate(self):
        """
        Alias for validate
        """
        self.validate()

    def test(self):
        """
        Runs single epoch while not updating any parameters
        Should only be run once after training is complete
        """
        
        self.model.eval() #turn off do/bn
        self.mode = 'Testing'
        loss, acc = self._iteration(self.test_dataloader) #run single pass
        self.losses['test_loss'].append(loss) #append losses
        self.losses['test_acc'].append(acc) #append accuracy

    def _iteration(self, dataloader):
        """
        Runs single epoch of forward and backward passes through the data

        Gets data (as Tensors) from dataloader
        Converts to variables
        Puts on GPU (if you have one)
        Zeros gradients from last param update (if training)
        Passes Variables through the model
        Calculates losses (sums losses for multi-output models)
        Backward pass to get gradients (if training)
        Clips parameters (if training)
        Updates parameters (if training)
        Returns loss averaged over epoch
        """

        epoch_loss = 0
        epoch_acc = 0

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
                loss = sum([self.criterion(_y_pred, _y.squeeze()) for _y_pred, _y in zip(y_pred, y)])
            else:
                loss = self.criterion(y_pred, y[0].squeeze())

                _, pred = torch.max(y_pred.data, 1)
                correct = (pred == y[0].squeeze().data)
                epoch_acc += correct.sum()/len(correct)

            if self.mode == 'Training':
                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)
                self.optimizer.step()

            epoch_loss += loss.data[0]

        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

    def save(self, path):
        """
        Saves model parameters to path
        """
        torch.save(self.model.state_dict(), path)



        
    
        