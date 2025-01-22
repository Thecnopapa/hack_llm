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

    def __init__(self, data_frame, transform=None):
        cols = list(data_frame.columns)
        ordered_cols = ["timestamp"]
        for c in cols:
            if not ("time" in c or  "Unnamed" in c):
                ordered_cols.append(c)
        self.df = data_frame[ordered_cols].sort_values("timestamp", ascending=True)
        self.transform = transform


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
        sample = {'timestamp': timestamp, 'values': values.reshape(1,5), "time":time.reshape(1,9)} #.reshape(-1)
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
    def __call__(self, sample):
        timestamp, values, time = sample['timestamp'], sample['values'], sample['time']
        return {'timestamp': timestamp, 'values': torch.from_numpy(values), "time": torch.from_numpy(time)}




def show_values_batch(sample_batched):
    time, timestamps, values_batch = sample_batched['time'], sample_batched["timestamp"], sample_batched['values']
    batch_size = len(timestamps)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(batch_size):
        ax.bar(int(timestamps[i]), values_batch[i])
        plt.title('Batch from dataloader')
    fig.show()





# For testing:
if __name__ == '__main__':

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




