import torch.optim as optim

#these are just aliases for the PyTorch optimizers
#this is done so the user only has to `import blib`, not `import torch` too

Adadelta = optim.Adadelta
Adagrad = optim.Adagrad
Adamax = optim.Adamax
Adam = optim.Adam
RMSprop = optim.RMSprop
RMSProp = optim.RMSprop #alias
SGD = optim.SGD