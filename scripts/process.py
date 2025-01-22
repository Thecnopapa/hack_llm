# Essential
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Other scripts
from utilities import *



##### .csv to pd.DataFrame #####


# Define time origin, used to measure time between entries
origin = "1970-01-01T00:00:00"
origin_datetime = datetime.fromisoformat(origin)


# Remove rows with missing values
def filter_nas(data):
    data["has_null"] = data.isnull().any(axis=1)
    data = data[data["has_null"] == False]
    return data


# Normalise NO2 between 0 and 1
def normalise(data):
    data["NO2"] = data["NO2"] /data["NO2"].abs().max()
    return data


# Add a leading 0 to the hour when necessary and convert to string
def hour_add_0(row):
    hour = row["hour"]-1
    row["hour"] = "{}{}".format( len(str(hour)) == 1 and '0' or '', hour)
    return row


# Merge the date and hour into a single string on a new column
def format_time(data):
    data = data.apply(hour_add_0, axis=1)
    data["time"] = data.data + "T" + data.hour + ":00:00"
    return data


# Save the total hours of each date/time as a new column
def calculate_timestamp(data):
    #time = datetime.fromisoformat(time)
    data["timestamp"] = data["time"].apply(lambda x: get_total_hours(x))
    return data


# Calculate the total hour of the timestamp since 1970/01/01
def get_total_hours(timestamp):
    return (datetime.fromisoformat(timestamp)-origin_datetime).total_seconds() / 3600


# Return 2 datasets:  1 week before a defined date/time & 24h after the same date/time
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


# Transform the dataframe so that the data is grouped entry (in time) / Very slow
def merge_entries(data, is_one_station = True):
    columns = ["time"]
    if "nom_estacion" in data.columns:
        unique_stations = sorted(data["nom_estacio"].unique())
        number_of_stations = len(unique_stations)
        for station in unique_stations:
            columns.append(station)
        is_one_station = False
    else:
        columns.append("NO2")
    merged = pd.DataFrame(columns=columns)
    unique_hours = data["time"].unique()
    progress = ProgressBar(len(unique_hours))
    for time in unique_hours:
        subset = data[data["time"] == time]
        if len(subset) == len(columns)-1:
            if len(subset) > 1:
                subset = subset.sort_values(by="nom_estacio", ascending=True)
            row_list = [time] + list(subset["NO2"].values)
            for i in range(len(row_list)):
                row_list[i] = [row_list[i]]
            row_dict = dict(zip(columns, row_list))
            new_row = pd.DataFrame(row_dict)
            merged = pd.concat([merged, new_row], ignore_index=True)
        progress.add()
    if is_one_station:
        return merged, ["NO2"]
    else:
        return merged, unique_stations

def df_by_station(data):
    stations = data["nom_estacio"].unique()
    dataframes = []
    for station in stations:
        station_data = data[data["nom_estacio"] == station]
        station_path = "../data/stations/station_{}.csv".format(station)
        station_data.to_csv(station_path, index=False)





# Process the data to a useful format either from a pd.Dataframe object or from a path
def process_data(data, name = "d", as_path=False):
    if as_path:
        data = pd.read_csv(data, na_values=np.nan, dtype={"NO2": float, "data": str})
    data = filter_nas(data)
    df_by_station(data)

    for station_df in os.listdir("../data/stations"):



    data = normalise(data)
    data = format_time(data)
    data, stations = merge_entries(data)
    data = calculate_timestamp(data)
    data.to_csv("../data/{}_processed.csv".format(name), index=False)
    return data, stations


# For testing:
if __name__ == '__main__':
    data = filter_nas("../data/trainData.csv")
    data = normalise(data)
    print(data)

    data = format_time(data)
    print(data)

    data, stations = merge_entries(data)
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


