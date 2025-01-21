import os

import torch
import pandas as pd
import numpy as np
from datetime import datetime


origin = "1970-01-01T00:00:00"
origin_datetime = datetime.fromisoformat(origin)

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
    hour = row["hour"]-1
    row["hour"] = "{}{}".format( len(str(hour)) == 1 and '0' or '', hour)
    return row


def format_time(data, time= origin):
    time = datetime.fromisoformat(time)
    data = data.apply(hour_add_0, axis=1)
    data["time"] = data.data + "T" + data.hour + ":00:00"
    #data["time"] = pd.to_datetime(data["time"])
    data["total_hours"] = data["time"].apply(lambda x: (time-datetime.fromisoformat(x)).total_seconds() / 3600)
    #data["time"] = data.time.transform(lambda x: x.replace(())
    print(len(data))
    return data


def get_total_hours(timestamp):
    return (origin_datetime-datetime.fromisoformat(timestamp)).total_seconds() / 3600

def get_window(data, time):

    hours = get_total_hours(time)
    one_week_less = hours - 168
    one_day_more = hours + 24
    print(one_week_less, one_day_more)
    week = data[data["total_hours"] <=hours]
    week =  week[week["total_hours"] >= one_week_less]
    day = data[data["total_hours"] > hours ]
    day = day[day["total_hours"] <= one_day_more]
    return  week, day



data = filter_nas("../data/trainData.csv")

data = format_time(data)
print(len(data))
print(data)
data.to_csv("../data/processedData.csv")




example_week, example_day= get_window(data, "2017-06-03T00:00:00")
example_week.to_csv("../data/exampleWeek.csv")
example_day.to_csv("../data/exampleDay.csv")
print(example_week)
print(example_day)


