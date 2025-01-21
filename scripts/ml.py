import os

import numpy.ma.extras
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


data_frame = pd.read_csv('../data/processedData.csv', index_col=False)


class PollutionDataset(Dataset):

    def __init__(self, data_frame, transform=None):
        cols = list(data_frame.columns)
        ordered_cols = ["timestamp"]
        for c in cols:
            if not ("time" in c or  "Unnamed" in c):
                ordered_cols.append(c)
        self.df = data_frame[ordered_cols].sort_values("timestamp", ascending=False)
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        timestamp = self.df.iloc[idx,0]
        values = self.df.iloc[idx,1:]
        values = np.array(values, dtype=float)
        sample = {'timestamp': timestamp, 'values': values}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def plot_values(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, sample in enumerate(self):
            print(i, sample["timestamp"], sample["values"])
            #plot_values(, , ax)
            ax.bar(int(sample["timestamp"]), numpy.mean(sample["values"]))
        plt.show()

class ToTensor(object):
    def __call__(self, sample):
        timestamp, values = sample['timestamp'], sample['values']
        return {'timestamp': timestamp, 'values': torch.from_numpy(values)}


data = PollutionDataset(data_frame)
tensors = PollutionDataset(data_frame, transform=ToTensor())

# data.plot_values()
print(data.df)




dataloader = DataLoader(data, batch_size=64, shuffle=False)





