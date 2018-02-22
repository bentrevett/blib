import sys
sys.path.insert(0,'..') #quick hack for now until it's an actual module!
import blib

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data

import numpy as np

X1_train = torch.ones(100, 100).long()
X2_train = torch.ones(100, 100).long()
y_train = torch.ones(100,).long()

X1_val = torch.ones(100, 100).long()
X2_val = torch.ones(100, 100).long()
y_val = torch.LongTensor(np.random.randint(2, size=(100,)))

train_dataset = blib.data.TwoOneDataset(X1_train, X2_train, y_train)
val_dataset = blib.data.TwoOneDataset(X1_val, X2_val, y_val)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

class TwoOneNeuralNetwork(nn.Module):
    """
    A simple neural network that goes from 2 inputs (of 100 dim) to 1 output (of 2 dim).
    """
    def __init__(self):
        super(TwoOneNeuralNetwork, self).__init__()
        
        self.fc = nn.Linear(100,2)
        
    def forward(self, x1, x2):
        
        return self.fc(x1.float())

model = TwoOneNeuralNetwork()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = blib.train.Trainer((train_data_loader, val_data_loader), model, optimizer, criterion, n_inp=2)

for i in range(5):
    trainer.train()
    trainer.validate()
trainer.test()

print(trainer.losses)