import torch

class OneOnePadCollate:
    """
    This automatically pads variable length sequences within a batch to the maximum length sequence within that match
    Use by passing collate_fn=TwoOnePadCollate() to the torch.utils.data.DataLoader
    
    THIS ASSUMES YOUR PADDING INDEX IS ZERO, IF NOT THEN YOU'RE INCORRECTLY PADDING

    This may be slow, need to test.
    Adapted from: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
    """

    def __init__(self, dim=0):
        self.dim=dim

    def pad_collate(self, batch):
        #find longest sequence
        input1_max_len = max(list(map(lambda x: x[0].shape[self.dim], batch)))

        #used to store the padded sequences before stacking
        _input1 = []

        #pad each to the maximum length within the batch
        for _, (input1, output1) in enumerate(batch):
            _input1.append(torch.cat((input1, torch.zeros([input1_max_len-input1.shape[self.dim]]).long()),dim=self.dim))
        
        #stack all
        #TODO: do we need the list + map here? isn't it already a list?
        input1 = torch.stack(list(map(lambda x: x, _input1)), dim=0)
        output1 = torch.stack(list(map(lambda x: x[1], batch)), dim=0)

        #return the batch
        return input1, output1

    def __call__(self, batch):
        return self.pad_collate(batch)

class TwoOnePadCollate:
    """
    This automatically pads variable length sequences within a batch to the maximum length sequence within that match
    Use by passing collate_fn=TwoOnePadCollate() to the torch.utils.data.DataLoader
    
    THIS ASSUMES YOUR PADDING INDEX IS ZERO, IF NOT THEN YOU'RE INCORRECTLY PADDING

    This may be slow, need to test.
    Adapted from: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
    """

    def __init__(self, dim=0):
        self.dim=dim

    def pad_collate(self, batch):
        #find longest sequence
        input1_max_len = max(list(map(lambda x: x[0].shape[self.dim], batch)))
        input2_max_len = max(list(map(lambda x: x[1].shape[self.dim], batch)))
        
        #used to store the padded sequences before stacking
        _input1 = []
        _input2 = []

        #pad each to the maximum length within the batch
        for _, (input1, input2, output1) in enumerate(batch):
            _input1.append(torch.cat((input1, torch.zeros([input1_max_len-input1.shape[self.dim]]).long()),dim=self.dim))
            _input2.append(torch.cat((input2, torch.zeros([input2_max_len-input2.shape[self.dim]]).long()),dim=self.dim))
        
        #stack all
        #TODO: do we need the list + map here? isn't it already a list?
        input1 = torch.stack(list(map(lambda x: x, _input1)), dim=0)
        input2 = torch.stack(list(map(lambda x: x, _input2)), dim=0)
        output1 = torch.stack(list(map(lambda x: x[2], batch)), dim=0)

        #return the batch
        return input1, input2, output1

    def __call__(self, batch):
        return self.pad_collate(batch)