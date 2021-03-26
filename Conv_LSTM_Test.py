# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:00:45 2021

@author: karlo
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import datetime




class Meteo_MODIS_ds(Dataset):
    def __init__(self, 
                 csv_path, 
                 raster_path, 
                 seq_len_in, 
                 seq_len_out, 
                 input_size,
                 train_end = "2000-11-01", 
                 val_start="2000-11-01", 
                 test_start="2001-01-01",
                 **kwargs):
        """
        Parameters
        ----------
        csv_path (string): path to the csv file
        raster_path (string): path to the raster files
        is_valid (int): 1 for valid & 0 for training
        """        
        self.tabular_data = pd.read_csv(csv_path)
        self.tabular_input = self.tabular_data[self.tabular_data.columns[2:]]
        self.dates = self.tabular_data[self.tabular_data.columns[0]].astype("datetime64[ns]")
        self.raster_data = np.load(raster_path, allow_pickle=True)
        self.y = self.tabular_data[self.tabular_data.columns[1]] # observed runoff column
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.input_size = input_size
        
        
        print (self.dates.index)

    def __len__(self):
        return self.dates.__len__() - (self.seq_len_in - 1)

    
    def __getitem__(self, index):
        ras_shape1, ras_shape2 = self.raster_data[:,1][0].shape # getting the raster shape 
        tabular_input = np.array(self.tabular_input).astype("float32") # selecting tabular trainign data
        rasters = np.vstack(self.raster_data[:,1]).astype("float32").reshape(-1, ras_shape1, ras_shape2 ) # reshaping the raster 
        y =  np.array(self.y).astype("float32") # converting 
        # prepairing index + seq_len_in data
        tabular_input = tabular_input[index : index + seq_len_in]
        ras_data = rasters[index : index + seq_len_in]
        y_data =y[index + seq_len_in -1: index + seq_len_in + seq_len_out-1]
        
        data = {
            "raster": torch.tensor(ras_data),
            "tabular_data": torch.tensor(tabular_input, dtype=torch.float32),
            "runoff": torch.from_numpy(y_data),
            }
        
        return data

    # def train_val_dataset(dataset, val_split=0.25):
    #     train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    #     datasets = {}
    #     datasets['train'] = Subset(dataset, train_idx)
    #     datasets['val'] = Subset(dataset, val_idx)
    #     return datasets
    
    

    
 # ====== STATICS ======
num_batches = 32                # how many timeserieses to be trained in one iteration
input_size = 9                  # number of features (stations) --> rain, temp, hum, etc. 
hidden_size = 30                # number of hidden neurons
num_layers = 2                  # number of LSTM layers
seq_len_in = 5                # length of the training time series
seq_len_out = 1                 # number of output step for predicted runoff
# train_end = "2013-01-01"        # end date of training period
# val_start = "2013-01-01"        # start date of validation period
# test_start = "2014-01-01"       # testing period start date
# =====================   
    
train_data = Meteo_MODIS_ds(csv_path="input_data/Drava_data.csv",
                            raster_path="input_data/test_snow_data.npy",
                            seq_len_in = seq_len_in,
                            seq_len_out= seq_len_out,
                            input_size = input_size,)

training_dataloader = DataLoader(train_data,
                        batch_size=5,
                        drop_last=True,)
print ("The legth of the dataloader is:", len(training_dataloader))





if __name__ == "__main__":
    dataset = next(iter(training_dataloader))   
    print ("y_data shape:", dataset["runoff"].shape)
    print ("tabular input data shape:", dataset["tabular_data"].shape)
    print ("raster data input shape:", dataset["raster"].shape)
    
       
    print ("y_data shape:", dataset["runoff"])
    print ("tabular input data shape:", dataset["tabular_data"])

        