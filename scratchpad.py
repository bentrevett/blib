import blib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data

X_train = torch.ones(100_000, 100)
y_train = torch.zeros(100_000, 1)

X_val = torch.ones(100_000, 100)
y_val = torch.zeros(100_000, 1)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.fc = nn.Linear(100,1)
        
    def forward(self, x):
        
        return self.fc(x)

model = NeuralNetwork()

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

trainer = blib.train.VanillaTrainer(model, criterion, optimizer)

trainer.cpu()
trainer.train()

for batch, (prediction, loss) in trainer(train_data_loader):
    pass

trainer.eval()

for batch, (prediction, loss) in trainer(val_data_loader):
    pass

