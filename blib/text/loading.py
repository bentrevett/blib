# this has all the functions for loading data

import pandas as pd #from_csv

import random #shuffling

import glob #from_folders
import os #from_folders

def from_csv(path, cols=[0,1], skip_header=False, skip_rows=0, splits=None, shuffle=False, n_fields=None):
    """
    path -> path to csv file
    cols -> which columns of the csv file to load
    skip_header -> if the csv file has a header, skip it
    skip_rows -> number of rows to skip
    splits -> can split the data into train/val/test if it all exists in a single file
    shuffle -> whether to shuffle the data for the splits
    n_fields -> if you just want to use cols[0,n] set n_fields to n, overwrites cols
    """
    if shuffle==True:
        assert splits is not None, "You only shuffle here when you split the data, if you're looking to shuffle data passed in the neural network, this is NOT when you do it"

    if n_fields is not None:
        cols = list(range(n_fields))

    csv_data = pd.read_csv(path, header=None)

    data = []

    skip_header = 1 if skip_header else 0

    for i, _ in enumerate(csv_data):
        if i in cols:
            data.append(csv_data[i].values.tolist()[skip_header+skip_rows:])

    if shuffle == True:
        data = list(zip(*data))
        random.shuffle(data)
        data = list(zip(*data))
        for i, _ in enumerate(data): #back into list for consistency w/ unshuffled input
            data[i] = list(data[i])

    if splits is not None:
        assert len(splits) < 4 and len(splits) > 1, "Can only split 2 or 3 ways currently"
        assert sum(splits) == 1, "Your splits must sum to 1 currently"
        
        split_data = []
        
        if len(splits) == 2:
            split_idx = int(len(data[0])*splits[0])

            for d in data:
                split_data.append(d[:split_idx])

            for d in data:
                split_data.append(d[split_idx:])

            data = split_data

        else: #should only be len(splits) == 3
            assert len(splits) == 3

            first_split_idx = int(len(data[0])*splits[0])
            second_split_idx = int(len(data[0])*splits[1])

            for d in data:
                split_data.append(d[:first_split_idx])
            
            for d in data:
                split_data.append(d[first_split_idx:first_split_idx+second_split_idx])

            for d in data:
                split_data.append(d[first_split_idx+second_split_idx:])

            data = split_data

    return data

def from_folders(path, folders, shuffle=False, splits=None):
    """
    Stolen from: https://github.com/fastai/fastai/blob/e8433d4a76eaf40c29a74a18ee043b23f6c35dbe/fastai/text.py

    Changed it so labels are the text (folder name) and not numerical to make consistent w/ rest of the library
    """
    if shuffle==True:
        assert splits is not None, "You only shuffle here when you split the data, if you're looking to shuffle data passed in the neural network, this is NOT when you do it"

    texts, labels = [],[]

    for _, label in enumerate(folders):
        for fname in glob.glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(label)

    data = [texts, labels]

    if shuffle == True:
        data = list(zip(*data))
        random.shuffle(data)
        data = list(zip(*data))
        for i, _ in enumerate(data): #back into list for consistency w/ unshuffled input
            data[i] = list(data[i])
    
    if splits is not None:
        assert len(splits) < 4 and len(splits) > 1, "Can only split 2 or 3 ways currently"
        assert sum(splits) == 1, "Your splits must sum to 1 currently"

        split_data = []
        
        if len(splits) == 2:
            split_idx = int(len(data[0])*splits[0])

            for d in data:
                split_data.append(d[:split_idx])

            for d in data:
                split_data.append(d[split_idx:])

            data = split_data

        else: #should only be len(splits) == 3
            assert len(splits) == 3

            first_split_idx = int(len(data[0])*splits[0])
            second_split_idx = int(len(data[0])*splits[1])

            for d in data:
                split_data.append(d[:first_split_idx])
            
            for d in data:
                split_data.append(d[first_split_idx:first_split_idx+second_split_idx])

            for d in data:
                split_data.append(d[first_split_idx+second_split_idx:])

            data = split_data

    return data




