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
    if not os.path.exists("../model/model.pth") or force_train:
        process_data("../data/trainData.csv", as_path=True, force=force_process)
        dataloaders = create_dataloaders()
        model = TinyModel(24)
        trainTinyModel(dataloaders, model)
        save_model(model)

    else:
        model = load_model()


    model.eval()
    print("windowDara:",windowData)
    windowData = process_window(windowData)
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
    return pred.tolist()

# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../scripts/validationData.csv")
    predict(windowData[0:168], force_train = True, force_process=False)