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
from sklearn.preprocessing import MinMaxScaler



class Meteo_MODIS_ds(Dataset):
    def __init__(self, 
                 csv_path, 
                 raster_path, 
                 seq_len_in, 
                 seq_len_out, 
                 input_size,
                 date_start=None,
                 date_end = None,
                 **kwargs,
                 ):
        """
        Parameters
        ----------
        csv_path (string): path to the csv file
        raster_path (string): path to the raster files
        is_valid (int): 1 for valid & 0 for training
        """        
        self.tabular_data = pd.read_csv(csv_path, parse_dates=True,)
        self.tabular_input = self.tabular_data[self.tabular_data.columns[2:]]
        self.dates = pd.to_datetime(self.tabular_data[self.tabular_data.columns[0]], format="%d/%m/%Y")
        self.raster_data = np.load(raster_path, allow_pickle=True)
        self.y = self.tabular_data[self.tabular_data.columns[1]] # observed runoff column
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.input_size = input_size

        if date_start != None: # chaning starting date, if provided
            if pd.Timestamp(datetime.datetime.strptime(date_start, "%Y-%m-%d")) in (self.dates).values:
                self.date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d")
                print ("Starting date is valid.\nStarting day set to {}."\
                        .format(self.date_start.strftime('%d.%m.%Y')))
            else:
                raise Exception("The provided start date is not in range!")
        else: # setting start date to first value in date
            self.date_start = self.dates.iloc[0]
            print ("Starting date was not provided.\nFirst day of dataset ({}) was set as starting date."\
                    .format(self.date_start.strftime('%d.%m.%Y')))

        if date_end != None: # chaning end date, if provided
            if pd.Timestamp(datetime.datetime.strptime(date_end, "%Y-%m-%d")) in (self.dates).values:
                self.date_end = datetime.datetime.strptime(date_end, "%Y-%m-%d")
                print ("Ending date is valid.\nEnd day set to {}."\
                        .format(self.date_end.strftime('%d.%m.%Y')))
            else:
                raise Exception("The provided end date is not in range!")
        else: # setting end date to last value in date
            self.date_end = self.dates.iloc[-1]
            print ("Ending date was not provided.\nLast day of dataset ({}) was set as end date."\
                    .format(self.date_end.strftime('%d.%m.%Y')))    

        
    
    
    
    def __len__(self):
        return self.dates.__len__() - (self.seq_len_in - 1)

    def __getitem__(self, index):
        ras_shape1, ras_shape2 = self.raster_data[:,1][0].shape # getting the raster shape 
        tabular_input = np.array(self.tabular_input).astype("float32") # selecting tabular trainign data
        rasters = torch.tensor(np.vstack(self.raster_data[:,1]).astype("float32").reshape(-1, ras_shape1, ras_shape2 )) # reshaping the raster            
        y =  np.array(self.y).astype("float32") # converting 
        y_scaler = MinMaxScaler()
        tab_scaler = MinMaxScaler()

        tabular_scaled = tab_scaler.fit_transform(tabular_input) # rescaling [0, 1]
        _ = y_scaler.fit_transform(y.reshape(-1,1)) # rescaling [0, 1]
        rasters_scaled = rasters/rasters.unique().max() 
        """Rasters are divided by largest unique value (label) in order to get labels
        between 0 and 1 for faster training anf better results."""

        data = {
            "raster": rasters_scaled,
            "tabular_data": torch.tensor(tabular_input, dtype=torch.float32),
            "runoff": torch.from_numpy(y),
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
train_start = "2000-04-02"
train_end = "2000-08-31"        # end date of training period
val_start = "2000-09-01"        # start date of validation period
val_end = "2000-11-30"
test_start = "2000-12-01"       # testing period start date
test_end = "2001-01-12"
# =====================   
    
train_data = Meteo_MODIS_ds(csv_path="input_data/Drava_data.csv",
                            raster_path="input_data/test_snow_data.npy",
                            
                            seq_len_in = seq_len_in,
                            seq_len_out= seq_len_out,
                            input_size = input_size,
                            date_start = "2000-04-02",
                            date_end="2001-01-10"
                            )

training_dataloader = DataLoader(train_data,
                        batch_size=5,
                        drop_last=True,)
print ("The legth of the dataloader is:", len(training_dataloader))





if __name__ == "__main__":
    dataset = next(iter(training_dataloader))   
    # print ("y_data shape:", dataset["runoff"].shape)
    print ("tabular input data shape:", dataset["tabular_data"].shape)
    print ("raster data input shape:", dataset["raster"].shape)
    
       
    # print ("y_data shape:", dataset["runoff"])
    # print ("tabular input data shape:", dataset["tabular_data"])
    print ("raster data input shape:", dataset["raster"][0][0])
        