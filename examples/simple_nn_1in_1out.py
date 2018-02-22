import sys
sys.path.insert(0,'..') #quick hack for now until it's an actual module!
import blib

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data

import numpy as np

X_train = torch.ones(100, 100)
y_train = torch.ones(100, 1).long()

X_val = torch.ones(100, 100)
y_val = torch.LongTensor(np.random.randint(2, size=(100,)))

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

class OneOneNeuralNetwork(nn.Module):
    """
    A simple neural network that goes from 1 input (of 100 dim) to 1 output (of 2 dim).
    """
    def __init__(self):
        super(OneOneNeuralNetwork, self).__init__()

        self.fc = nn.Linear(100,2)
        
    def forward(self, x):
        
        return self.fc(x)

model = OneOneNeuralNetwork()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = blib.train.Trainer((train_data_loader, val_data_loader), model, optimizer, criterion)

for i in range(5):
    trainer.train()
    trainer.validate()
trainer.test()

print(trainer.losses)