import blib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data

import numpy as np

X_train = torch.ones(100_000, 100)
y_train = torch.LongTensor(np.random.randint(2, size=(100_000,)))
#y_train = torch.zeros(100_000).long()

X_val = torch.ones(100_000, 100)
y_val = torch.zeros(100_000).long()

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

acc_fn = blib.util.metrics.CategoricalAccuracy()
f1_fn = blib.util.metrics.F1Score(positive_label=0)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.fc = nn.Linear(100,2)
        
    def forward(self, x):
        
        return self.fc(x)

model = NeuralNetwork()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = blib.train.VanillaTrainer(model, criterion, optimizer)

trainer.cpu()
trainer.train()

for (batch_X, batch_y), (predictions, loss) in trainer(train_data_loader):
    pass
    
trainer.eval()

for batch, (predictions, loss) in trainer(val_data_loader):
    pass

