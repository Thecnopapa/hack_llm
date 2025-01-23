# Essential
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def hours_to_datetime(hours):
    return origin_datetime + timedelta(hours=hours)

# Save the total hours of each date/time as a new column
def calculate_timestamp(data):
    #time = datetime.fromisoformat(time)
    data["timestamp"] = data["time"].apply(lambda x: get_total_hours(x))
    data["month"] = data["timestamp"].apply(lambda x: hours_to_datetime(x).month)
    data["day"] = data["timestamp"].apply(lambda x: hours_to_datetime(x).day)
    data["hour"] = data["timestamp"].apply(lambda x: hours_to_datetime(x).hour)
    return data



# Calculate the total hour of the timestamp since 1970/01/01
def get_total_hours(timestamp):
    return int((datetime.fromisoformat(timestamp)-origin_datetime).total_seconds() / 3600)




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
    for station in stations:
        station_data = data[data["nom_estacio"] == station]
        station_path = "../data/stations/{}.csv".format(station)
        os.makedirs(os.path.dirname(station_path), exist_ok=True)
        station_data.to_csv(station_path, index=False)



# Process the data to a useful format either from a pd.Dataframe object or from a path
def process_data(data, name = "d", as_path=False, force= False):
    print("\nProcessing data...")
    if as_path:
        data = pd.read_csv(data, na_values=np.nan, dtype={"NO2": float, "data": str})
    data = filter_nas(data)
    df_by_station(data)

    for station in os.listdir("../data/stations"):
        print(station)
        if not os.path.exists("../data/processed/{}".format(station)) or force:
            station_df = pd.read_csv("../data/stations/{}".format(station))
            station_df = normalise(station_df)
            station_df = format_time(station_df)
            station_df = calculate_timestamp(station_df)
            station_df = station_df[["timestamp", "NO2", "month", "day", "hour"]]
            os.makedirs("../data/processed", exist_ok=True)
            station_df.to_csv("../data/processed/{}".format(station), index=False)
    print("Processed {} stations".format(len(data.columns.unique())))

# For testing:
if __name__ == '__main__':
    process_data("../data/trainData.csv", as_path=True, force=True)



