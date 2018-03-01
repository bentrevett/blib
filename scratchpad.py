import blib

blib.utils.set_seed(1234)

train_path = '../aclImdb_v1/train'
test_path = '../aclImdb_v1/test'
folders = ['neg', 'pos']

X_train, y_train, _, _ = blib.text.from_folders(train_path, folders, splits=[0.01,0.99])
X_test, y_test, _, _ = blib.text.from_folders(test_path, folders, splits=[0.01, 0.99])

#vocab source must always be wrapped in a list, even if there is only one!
#X_vocab = blib.text.build_vocab([X_train], max_size=100, min_freq=2, tokenizer='spacy')

#X_train, X_val, X_test = blib.text.tokenize(X_vocab, [X_train, X_val, X_test])

X_vocab, X_train, X_test = blib.text.build_and_tokenize([X_train, X_test], max_size=20_000, max_length=100, tokenizer='spacy')

y_vocab, y_train, y_test = blib.text.build_and_tokenize([y_train, y_test], unk_token=None, pad_token=None)

train_dataset = blib.data.OneOneDataset(X_train, y_train)

train_dataloader = blib.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=blib.data.OneOnePadCollate())

test_dataset = blib.data.OneOneDataset(X_test, y_test)

test_dataloader = blib.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=blib.data.OneOnePadCollate())

emb_dim = 64
hid_dim = 64
rnn_type = 'LSTM'
n_layers = 1
bidirectional = True
dropout = 0.5

model = blib.models.RNNClassification(len(X_vocab),
                                      len(y_vocab),
                                      emb_dim,
                                      hid_dim,
                                      rnn_type,
                                      n_layers,
                                      bidirectional,
                                      dropout)


#opt and loss are just convenience wrappers so you only need the blib include
optimizer = blib.opt.Adam(model.parameters())
criterion = blib.loss.CrossEntropy()

trainer = blib.train.Trainer([train_dataloader, test_dataloader], model, optimizer, criterion)

n_epochs = 10
patience = 1

trainer.run(n_epochs, patience)
