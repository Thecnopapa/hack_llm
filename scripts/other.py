import os

import torch
import pandas as pd
import numpy as np
from datetime import datetime
# No se si va
def filter_nas(data_path):
    data = pd.read_csv(data_path, na_values=np.nan, dtype={"NO2":float, "data":str})
    print(len(data))
    data["has_null"] = data.isnull().any(axis=1)
    #print(null_data)
    '''index = 0
    for nan in null_data:
        if nan:
            #print("dropping")
            data.drop(index=index, inplace=True)
        index+=1'''
    data = data[data["has_null"] == False]
    print(len(data))

    return data
def hour_add_0(row):
    hour = row["hour"]
    row["hour"] = "{}{}".format( len(str(hour)) == 1 and '0' or '', hour)
    return row


def format_time(data, current_time):

    data = data.apply(hour_add_0, axis=1)
    data["time"] = data.data + "T00:" + data.hour + ":00"
    #data["time"] = pd.to_datetime(data["time"])
    data["total_hours"] = data["time"].apply(lambda x: (datetime.fromisoformat(x) - current_time).total_seconds() / 3600)
    #data["time"] = data.time.transform(lambda x: x.replace(())
    print(len(data))

    return data

def select_week(data):
    data = data[data["total_hours"] <=0]
    data =  data[data["total_hours"] >= -168]
    return  data


data = filter_nas("../data/trainData.csv")
current_time = datetime.fromisoformat("2017-06-03T00:00:00")
data = format_time(data, current_time)
data = select_week(data)
print(len(data))
print(data)
data.to_csv("../data/processedData.csv")