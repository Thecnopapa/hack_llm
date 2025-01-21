import os

import torch
import pandas as pd
import numpy as np

# No se si va
def filter_nas(data_path):
    data = pd.read_csv(data_path, na_values=np.nan, dtype={"NO2":float})
    print(len(data))
    null_data = data.isnull().any(axis=1)
    #print(null_data)
    index = 0
    for nan in null_data:
        if nan:
            #print("dropping")
            data.drop(index=index, inplace=True)
        index+=1
    print(len(data))
    data.to_csv("../data/processedData.csv")
    return data
