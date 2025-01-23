# Essential
import os
import pandas as pd
import numpy as np



# Other scripts
from process import process_data, process_window
from dataload import create_dataloaders, week_to_X
from model import customModel
from train import train


##### Our solution #####


# To store processed training data

# Our predict function


def predict(windowData, force_train=False):
    process_data("../data/trainData.csv", as_path=True, force=False)
    dataloaders = create_dataloaders()
    model = customModel()
    train(dataloaders, model)
    window = process_window(windowData)
    window_X = week_to_X(window)
    pred = model(window_X)
    print(pred)

# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../validationData.csv")
    predict(windowData)