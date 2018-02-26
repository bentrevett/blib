import sys
sys.path.insert(0,'..') #quick hack for now until it's an actual module!
import blib

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data

X1_train = [[2, 4, 6, 8], [2, 4, 6], [2, 4], [2]]
X2_train = [[1,3,5,7],[1],[3,5,7],[9]]
y1_train = [0, 1, 0, 1]

class VariableLengthSequences(nn.Module):
    def __init__(self):
        super(VariableLengthSequences, self).__init__()

        self.embedding = nn.Embedding(50, 100)
        self.rnn = nn.GRU(100, 256)
        self.fc = nn.Linear(256, 2)

    def forward(self, x1, x2):

        x = self.embedding(x1)
        x = x.permute(1, 0, 2)
        _, h = self.rnn(x)
        h = h.squeeze(0)
        return self.fc(h)

model = VariableLengthSequences()

train_dataset = blib.data.TwoOneDataset(X1_train, X2_train, y1_train)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=blib.data.TwoOnePadCollate())

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

#data loader must always be wrapped in a list, even if there is only one!
trainer = blib.train.Trainer([train_data_loader], model, optimizer, criterion, n_inp=2)

for i in range(5):
    trainer.train()
    trainer.validate()
trainer.test()

print(trainer.losses)