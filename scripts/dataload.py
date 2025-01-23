# Essentials
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotting
import matplotlib.pyplot as plt

# Pytorch-related
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Other scripts
from process import origin_datetime

# Ignore warnings (idk why)
import warnings
warnings.filterwarnings("ignore")



##### pd.DataFrame to torch.Dataset #####


def hours_to_datetime(hours):
    return origin_datetime + timedelta(hours=hours)


class PollutionDataset(Dataset):

    def __init__(self, data_frame, transform=None, is_test=False, output_dim = 1):
        cols = list(data_frame.columns)
        ordered_cols = ["timestamp"]
        for c in cols:
            if not ("time" in c or  "Unnamed" in c):
                ordered_cols.append(c)
        self.df = data_frame[ordered_cols].sort_values("timestamp", ascending=True)
        self.transform = transform
        self.is_test = is_test
        self.output_dim = output_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        timestamp = self.df.iloc[idx,0]
        time = hours_to_datetime(timestamp)
        time = np.array(time.timetuple(), dtype=np.float32)
        #time = np.array(list(map(float, time)))
        values = self.df.iloc[idx,1:]
        values = np.array(values, dtype=float)
        if self.output_dim == 1:
            sample = {'timestamp': timestamp, 'values': np.mean(values).reshape(1,1), "time":time.reshape(1,9)} #.reshape(-1)
        else:
            sample = {'timestamp': timestamp, 'values': values.reshape(1,self.output_dim), "time":time.reshape(1,9)} #.reshape(-1)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def plot_values(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, sample in enumerate(self):
            #print(i, sample["timestamp"], sample["values"])
            #plot_values(, , ax)
            ax.bar(int(sample["timestamp"]), numpy.mean(sample["values"]))
        plt.show()

class ToTensor(object):
    def __call__(self, array):
        return torch.from_numpy(array).to(torch.float)
        #for key in sample.keys():
        #    sample[key] = torch.from_numpy(sample[key])
        #return sample
        #timestamp, values, time = sample['timestamp'], sample['values'], sample['time']
        #return {'timestamp': timestamp, 'values': torch.from_numpy(values), "time": torch.from_numpy(time)}




def show_values_batch(sample_batched):
    time, timestamps, values_batch = sample_batched['time'], sample_batched["timestamp"], sample_batched['values']
    batch_size = len(timestamps)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(batch_size):
        ax.bar(int(timestamps[i]), values_batch[i])
        plt.title('Batch from dataloader')
    fig.show()


# Return 2 datasets:  1 week before a defined date/time & 24h after the same date/time
def get_window(data, hours):

    one_week_less = hours - 168
    one_day_more = hours + 24
    #print(one_week_less, one_day_more)
    week = data[data["timestamp"] < hours]
    week =  week[week["timestamp"] >= one_week_less]
    day = data[data["timestamp"] >= hours ]
    day = day[day["timestamp"] < one_day_more]
    return  week, day


def week_to_X(week, threshold, transform = None):
    #timestamps = np.array(threshold - week["timestamp"].values)
    values = np.array(week["NO2"].values)
    #array = np.array([timestamps,values], dtype=np.float64)
    array = values
    if transform is not None:
        array = transform(array)
    return array

def day_to_y(day, threshold, transform = None):
    #timestamps = np.array(threshold + day["timestamp"].values)
    values = np.array(day["NO2"].values)
    #array = np.array([timestamps,values], dtype=np.float64)
    array = values
    if transform is not None:
        array = transform(array)
    return array


class customDataLoader():
    def __init__(self, df, transform=None, name="dataset", is_train = True):
        self.name = name
        #print("\n",self.name, self)
        self.transform = transform
        self.df = df.sort_values("timestamp", ascending=True)
        self.length = len(self.df)
        self.start = int(min(self.df["timestamp"])+168)
        self.end = int(max(self.df["timestamp"]))
        self.days = self.len = int((self.end-self.start)//24)
        self.weeks = int(self.days//7)
        self.shift = 0
        #print(" - number of days:", self.days)
        #print(" - number of weeks:", self.weeks)
        #print(" - range:", self.start, self.end)

        self.missing_data = 0
        self.complete_data = 0

        self.curate()

    def curate(self):
        for i in range(self.days):
            threshold = self.start + i * 24
            week, day = get_window(self.df, threshold)
            if len(week) != 168 or len(day) != 24:
                self.missing_data += 1
            else:
                self.complete_data += 1
            print(i, end="\r")
        #print(" - missing data:", self.missing_data)
        #print(" - complete data:", self.complete_data)
        self.len = self.complete_data


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        threshold = self.start + (idx + self.shift) * 24
        week, day = get_window(self.df, threshold)
        if len(week) != 168 or len(day) != 24:
            self.shift+=1
            return self[idx]
        else:
            return (week_to_X(week, threshold, self.transform),
                      day_to_y(day, threshold, self.transform))

    def __iter__(self):
        for i in range(len(self)):
            yield i, self[i]
        self.shift = 0







def create_dataloaders():
    print("\nCreating dataloaders...")
    dataloaders = []
    for station in os.listdir("../data/processed"):
        path = f"../data/processed/{station}"
        dataloaders.append(customDataLoader(pd.read_csv(path), transform=ToTensor(), name=station))

    return dataloaders


# For testing:
if __name__ == '__main__':

    dataloaders = create_dataloaders()





'''


    quit()
    data_frame = pd.read_csv('../data/processedData.csv', index_col=False)

    data = PollutionDataset(data_frame)
    tensors = PollutionDataset(data_frame, transform=ToTensor())

    testTensors = PollutionDataset(data_frame, transform=ToTensor())

    # data.plot_values()
    print(data.df)

    print(tensors[124])

    dataloader = DataLoader(tensors, batch_size=1, shuffle=False, num_workers=0)

    print(len(dataloader))







    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['time'].size(),
              sample_batched['values'].size())

        # observe 4th batch and stop.
        #show_values_batch(sample_batched)
        if i_batch == 3:
            break



'''
