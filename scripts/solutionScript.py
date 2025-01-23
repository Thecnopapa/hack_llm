# Essential
import os
import pandas as pd
import numpy as np



# Other scripts
from process import process_data
from dataload import create_dataloaders
from model import customModel
from train import train


##### Our solution #####


# To store processed training data

# Our predict function


def predict(windowData, station = "average", output_dim=1):
    process_data("../data/trainData.csv", as_path=True, force=False)
    dataloaders = create_dataloaders()
    model = customModel()
    train(dataloaders, model)

# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../validationData.csv")
    predict(windowData)