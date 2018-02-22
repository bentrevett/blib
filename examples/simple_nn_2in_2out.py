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
y1_train = torch.ones(100,).long()
y2_train = torch.ones(100,).long()

X1_val = torch.ones(100, 100).long()
X2_val = torch.ones(100, 100).long()
y1_val = torch.LongTensor(np.random.randint(2, size=(100,)))
y2_val = torch.LongTensor(np.random.randint(2, size=(100,)))

train_dataset = blib.data.TwoTwoDataset(X1_train, X2_train, y1_train, y2_train)
val_dataset = blib.data.TwoTwoDataset(X1_val, X2_val, y1_val, y2_val)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

class TwoTwoNeuralNetwork(nn.Module):
    def __init__(self):
        super(TwoTwoNeuralNetwork, self).__init__()
        
        self.fc = nn.Linear(100,2)
        
    def forward(self, x1, x2):
        
        return self.fc(x1.float()), self.fc(x2.float())

model = TwoTwoNeuralNetwork()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = blib.train.Trainer((train_data_loader, val_data_loader), model, optimizer, criterion, n_inp=2, n_out=2)

for i in range(1):
    trainer.train()
    trainer.validate()
trainer.test()

print(trainer.losses)

