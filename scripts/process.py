import os

import torch
import pandas as pd
import numpy as np
from datetime import datetime
from utilities import *

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


def normalise(data):
    data["NO2"] = data["NO2"] /data["NO2"].abs().max()
    return data


def hour_add_0(row):
    hour = row["hour"]-1
    row["hour"] = "{}{}".format( len(str(hour)) == 1 and '0' or '', hour)
    return row


def format_time(data):
    data = data.apply(hour_add_0, axis=1)
    data["time"] = data.data + "T" + data.hour + ":00:00"
    return data

def calculate_timestamp(data):
    #time = datetime.fromisoformat(time)
    data["timestamp"] = data["time"].apply(lambda x: get_total_hours(x))
    return data

def get_total_hours(timestamp):
    return (datetime.fromisoformat(timestamp)-origin_datetime).total_seconds() / 3600

def get_window(data, time):
    hours = get_total_hours(time)
    one_week_less = hours - 168
    one_day_more = hours + 24
    print(one_week_less, one_day_more)
    week = data[data["timestamp"] <=hours]
    week =  week[week["timestamp"] >= one_week_less]
    day = data[data["timestamp"] > hours ]
    day = day[day["timestamp"] <= one_day_more]
    return  week, day


def merge_entries(data):
    unique_stations = sorted(data["nom_estacio"].unique())
    #print(unique_stations)
    number_of_stations = len(unique_stations)
    columns = ["time"]
    for station in unique_stations:
        columns.append(station)
    merged = pd.DataFrame(columns=columns)
    unique_hours = data["time"].unique()
    progress = ProgressBar(len(unique_hours))
    for time in unique_hours:
        subset = data[data["time"] == time]
        if len(subset) == number_of_stations:
            subset = subset.sort_values(by="nom_estacio", ascending=True)
            #print(subset)
            row_list = [time] + list(subset["NO2"].values)
            #print(row_list)
            for i in range(len(row_list)):
                row_list[i] = [row_list[i]]
            row_dict = dict(zip(columns, row_list))
            #print(row_dict)
            new_row = pd.DataFrame(row_dict)
            #print(new_row)
            merged = pd.concat([merged, new_row], ignore_index=True)
            #print(merged.loc[len(merged)-1].values)
        progress.add()
    return merged



if __name__ == '__main__':
    data = filter_nas("../data/trainData.csv")
    data = normalise(data)
    print(data)

    data = format_time(data)
    print(data)

    data = merge_entries(data)
    print(data)
    data.to_csv("../data/mergedData.csv")

data = pd.read_csv("../data/mergedData.csv", index_col=0)
data = calculate_timestamp(data)
print(data)

data.to_csv("../data/processedData.csv")









example_week, example_day= get_window(data, "2017-06-03T00:00:00")
example_week.to_csv("../data/exampleWeek.csv")
example_day.to_csv("../data/exampleDay.csv")
print(example_week)
print(example_day)


