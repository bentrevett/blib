# blib
**B**en's **Lib** for ML/DL utilities. 

Based off (i.e. some code copied from):
  - https://github.com/fastai/fastai/
  - https://github.com/dmarnerides/pydlt
  - https://github.com/mxbi/mlcrate

Everything based on the `trainer`, which is a rip-off of __fastai__'s `learner`.

Simlest way to use is:

`trainer = blib.train.Trainer((train_dataloader, val_dataloader, test_dataloader), model, optimizer, criterion)`

## TODO

  - Find a nice way to different metrics (such as accuracy and F1) to the `trainer`
  - Able to add multiple loss functions
  - Different loss function per output
  - More control over scheduler steps, i.e. step every batch as in SGDR