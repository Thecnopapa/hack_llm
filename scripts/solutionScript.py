import os

import torch
import pandas as pd
import numpy as np
from process import *



trainData = None # To store processed training data

def predict(windowData):
    # Load or create processed training data (slow) / First-run only
    global trainData
    if not "trainData_processed.csv" in os.listdir("../data"):
        print("Processed training data not found. Processing training data")
        trainData = process_data("../data/trainData.csv", name ="trainData", as_path=True)
        print("Training data processed and laoded")
    elif trainData is None:
        print("Loading processed training data")
        trainData = pd.read_csv("../data/trainData_processed.csv")
    else:
        print("Processed training data already loaded")

    # Process test dataframe (aka windowToPredict)
    print("Processing window data")
    windowData = process_data(windowData, name ="windowData")
    print(windowData)
    return




if __name__ == "__main__":
    windowData = pd.read_csv("../validationData.csv")
    predict(windowData)