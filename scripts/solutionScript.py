# Essential
import os
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch.utils.data import DataLoader


# Other scripts
from process import process_data
from dataload import PollutionDataset, ToTensor
from model import TinyModel, MeanModel
from train import trainModel, test


##### Our solution #####


# To store processed training data

# Our predict function
def predict(windowData, station = "average", output_dim=1):

# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../validationData.csv")
    predict(windowData)