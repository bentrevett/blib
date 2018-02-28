import blib

blib.utils.set_seed(1234)

path = '../aclImdb/test'
folders = ['neg', 'pos']

X_train, y_train, X_val, y_val, X_test, y_test = blib.text.from_folders(path, folders, shuffle=True, splits=[0.1,0.1,0.8])

#vocab source must always be wrapped in a list, even if there is only one!
#X_vocab = blib.text.build_vocab([X_train], max_size=100, min_freq=2, tokenizer='spacy')

#X_train, X_val, X_test = blib.text.tokenize(X_vocab, [X_train, X_val, X_test])

X_vocab, X_train, X_val, X_test = blib.text.build_and_tokenize([X_train], [X_train, X_val, X_test], max_size=100, min_freq=2, tokenizer='spacy')

y_vocab, y_train, y_val, y_test = blib.text.build_and_tokenize([y_train, y_val, y_test], unk_token=None, pad_token=None)

train_dataset = blib.data.OneOneDataset(X_train, y_train)

train_dataloader = blib.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=blib.data.OneOnePadCollate())

emb_dim = 256
hid_dim = 256
rnn_type = 'LSTM'
n_layers = 2
bidirectional = True
dropout = 0.5
n_epochs = 3

model = blib.models.RNNClassification(len(X_vocab),
                                      len(y_vocab))

import torch.nn as nn
import torch.optim as optim

#opt and loss are just convenience wrappers so you only need the blib include
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

trainer = blib.train.Trainer([train_dataloader], model, optimizer, criterion)

for e in range(1, n_epochs+1):
    trainer.train()
    trainer.validate()
trainer.test()