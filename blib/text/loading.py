# this has all the functions for loading data

import random
import pandas as pd

def from_csv(path, cols=[0,1], skip_header=False, skip_rows=0, splits=None, shuffle=False, n_fields=None):
    """
    path -> path to csv file
    cols -> which columns of the csv file to load
    skip_header -> if the csv file has a header, skip it
    skip_rows -> number of rows to skip
    splits -> can split the data into train/val/test if it all exists in a single file
    shuffle -> whether to shuffle the data or not
    n_fields -> if you just want to use cols[0,n] set n_fields to n, overwrites cols
    """
    
    if n_fields is not None:
        cols = list(range(n_fields))

    csv_data = pd.read_csv(path, header=None)

    data = []

    skip_header = 1 if skip_header else 0

    for i, _ in enumerate(csv_data):
        if i in cols:
            data.append(csv_data[i].values.tolist()[skip_header+skip_rows:])

    if splits is not None:
        #split_data = []
        assert sum(splits) == 1, "Your splits must sum to 1"
        
        #TODO: do splitting here after figuring out a clever way
        assert 1 == 2, 'SPLITTING NOT YET IMPLEMENTED'

    if shuffle == True:
        data = list(zip(*data))
        random.shuffle(data)
        data = list(zip(*data))
        for i, _ in enumerate(data): #back into list for consistency w/ unshuffled input
            data[i] = list(data[i])

    return data



