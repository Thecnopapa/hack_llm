# Essential
import os
import pandas as pd
import numpy as np

import torch

import scripts.model
import scripts.train
# Other scripts
from process import process_data, process_window
from dataload import create_dataloaders, week_to_X, ToTensor
from model import TinyModel
from train import trainTinyModel, test, save_model, load_model


##### Our solution #####


# To store processed training data

# Our predict function

n = 0
def predict(windowData, force_train=False, force_process=False):
    global n
    print(n,"/ ", end="")
    os.makedirs("../models", exist_ok=True)
    if len(os.listdir("../models")) == 0 or force_train:
        process_data("../data/trainData.csv", as_path=True, force=force_process)
        dataloaders = create_dataloaders()
        model = TinyModel(24)
        iterations = 5 # Up to 9 should be fine
        trainTinyModel(dataloaders, model, iterations=iterations)
    else:
        model = load_model()


    model.eval()
    print("windowDara:",windowData)
    windowData, min, max= process_window(windowData, normalise= True)
    print(windowData)
    threshold = max(windowData['timestamp'].values)-1
    window_X = week_to_X(windowData, threshold, transform=ToTensor())
    print("Input:")
    print(window_X)
    print(window_X.shape)
    print(window_X.dtype)

    pred = model(window_X)
    print("Prediction:")
    print(pred)
    print(pred.shape)
    n+=1
    pred = (pred * (max-min)) + min
    print("Rescaled:")
    print(pred)
    print(pred.shape)
    return pred.tolist()

# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../scripts/validationData.csv")
    predict(windowData[0:168], force_train = True, force_process=False)