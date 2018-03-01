# blib

**B**en's **Lib** for ML/DL utilities.

Based off (i.e. some code copied from):

- <https://github.com/fastai/fastai/>
- <https://github.com/dmarnerides/pydlt>
- <https://github.com/mxbi/mlcrate>
- <https://github.com/allenai/allennlp>


The goal is for something like:

- `train` has all of the wrapper functionality for PyTorch, including pre-build models
- `data` has all the PyTorch 'data' functionality, like datasets and dataloaders
- `datasets` has functionality to download and use actual datasets
- `text` has all of the text processing, i.e. formatting into csv, reading with pandas, vocab stuff
- `metrics` has all of the metrics
- `models` has pre-built models which can be wrapped with a `trainer`
- `utils` everything else

## How to use

### Loading data

``` python
#simple multi-csv -> list
X_train, y_train = blib.text.from_csv('data/data.train')
X_val, y_val = blib.text.from_csv('data/data.val')
X_test, y_test = blib.text.from_csv('data/data.test')

#can just get from individual fields, this gets first and fourth column
#can skip the first row, or skip_header=True to skip the first row
X_train, y_train = blib.text.from_csv('data/data.train', cols=[0,3], skip_rows=1)
X_train, y_train = blib.text.from_csv('data/data.train', cols=[0,3], skip_header=True) #same as above!

#n_fields argument gets the first n columns, easier do to n_fields=5 than cols=[0,1,2,3,4]
#n_fields currently doesn't have an offset, i.e. can't do cols 1,2,3,4,5
#if you put n_fields and cols, n_fields overwrites cols, so be careful!
X_train, y_train = blib.text.from_csv('data/data.train', cols=[0,1,2,3,4], skip_rows=1)
X_train, y_train = blib.text.from_csv('data/data.train', n_fields=5, skip_rows=1) #same as above!
X_train, y_train = blib.text.from_csv('data/data.train', cols=[1,2,3], n_fields=3, skip_rows=1) #gets columns 0,1,2 as n_fields overwrites cols!

#can also load data from folders/files, say we have /data/neg/example1.txt and /data/pos/example2.txt
X_train, y_train = blib.text.from_folders('data',['neg','pos'])

#simple single-csv -> list with splits and optional shuffling (note: shuffling happens before the split!)
#order must be: train_field_0, ..., train_field_n, val_field_0, ... val_field_n, etc.
X_train, y_train, X_val, y_val, X_test, y_test = blib.text.from_csv('data/data.all', splits=[0.7,0.15,0.15], shuffle = True)

#multi-in/out single-csv -> list
X1_train, X2_train, y_train, X1_val, X2_val, y_val = blib.text.from_csv('data/data.all', splits=[0.8,0.2], n_fields=3)
```

### Tokenizing data

``` python
#build the dictionary from sources
#NOTE: must be wrapped in a list!
X_vocab = blib.text.build_vocab([X_train])

#can be from multiple sources
#say you want to build the vocab from the train, validation and test sets
X_vocab = blib.text.build_vocab([X_train, X_val, X_test])

#default is to tokenize splitting text on spaces (use if data is already tokenized in the .csv)
#has spacy tokenizer built-in
X_vocab = blib.text.build_vocab([X1_train, X2_train], max_size=20_000, min_freq=3, max_length=100, pad=True, unk_token='<UnK>', pad_token='@PAD@', start_token='<SOS>', end_token='</SOS>', tokenizer='spacy')

#can also directly build the vocab object
#these two lines do the same one as above
X_vocab = blib.text.Vocab(max_size=20_000, min_freq=3, ...)
X_vocab.build_vocab([X1_train, X2_train])

#turn into tokens
X_train, X_val, X_test = blib.text.tokenize(X_vocab, [X_train, X_val, X_test])
X_train, X_val, X_test = X_vocab.tokenize([X_train, X_val, X_test]) #same as above

#this does the building vocab and tokenization in one step
#note how the first returned value is the vocab object
#this is how to build the vocab from one set of sources and tokenize another set
X_vocab, X_train, X_val, X_test = blib.text.build_and_tokenize([X_train], [X_train, X_val, X_test], max_size=20_000, min_freq=3)
#this is how to build and tokenize the vocab from one set of sources
y_vocab, y_train, y_val, y_test = blib.text.build_and_tokenize([y_train, y_val, y_test], unk_token=None, pad_token=None)
```

### Creating dataset and dataloader

``` python
#blib.data has PyTorch dataset objects for 1-2 inputs and 1-2 outputs
train_dataset = blib.data.OneOneDataset(X_train, y_train) #one input, one output
train_dataset = blib.data.TwoOneDataset(X1_train, X2_train, y_train) #two input, one output

#the blib dataloader is just a wrapper for the PyTorch one
#note: if you do not set `pad=True` when building the vocab, you need to pass the appropriate
#PadCollate object to the DataLoader, this handles padding within batches to the maximum length
#sequence within a batch
train_dataloader = blib.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=blib.data.TwoOnePadCollate())
```

### Using pre-build models

``` python
model = blib.models.RNNClassification(len(X_vocab), #input dim = input vocab size
                                      len(y_vocab), #output dim = output vocab size
                                      emb_dim,
                                      hid_dim,
                                      rnn_type, #LSTM, GRU or RNN
                                      n_layers,
                                      bidirectional, #True for bidirectional, False for unidirectional
                                      dropout,
                                      glove_dim, #not yet implemented
                                      freeze_emb) #not yet implemented
```

## Define optimizer loss function and off you go

``` python

#opt and loss are just convenience wrappers so you only need the blib import
optimizer = blib.opt.Adam(model.parameters())
scheduler = blib.opt.ReduceLROnPlateau(optimizer) #schedulers are in the opt
criterion = blib.loss.CrossEntropyLoss()
criterion = blib.loss.CrossEntropy() #also have all the PyTorch losses without the 'Loss' at the end
```

## Create the trainer and train

``` python
#wrap the dataloaders in a list
trainer = blib.train.Trainer([train_dataloader, val_dataloader, test_dataloader], model, optimizer, criterion)

#can pass where the load model parameters
#will save model every epoch if validation accuracy is lower than best validation accuracy
#clip the gradients to deal with exploding gradient problem
trainer = blib.train.Trainer([train_dataloader, val_dataloader, test_dataloader], model, optimizer, criterion, scheduler=scheduler, load_path='checkpoint.pt', save_path='new_checkpoint.pt', clip=0.5)

#call each epoch individually
for e in range(1, n_epochs+1):
    trainer.train()
    trainer.validate()
trainer.test()

#run for so many epochs
trainer.run(n_epochs)

#early stopping if validation loss doesn't improve after 3 epochs 
trainer.run(n_epochs, 3)

#early stopping if validation accuracy doesn't improve after 3 epochs
trainer.run(n_epochs, patience=3, patience_metric='val_acc', patience_mode='max')

#manually saving model
trainer.save(save_path)
```