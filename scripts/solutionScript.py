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
def predict(windowData, force_train=False, force_process=False, cheat=True):
    global n
    print(n,"/ ", end="")
    os.makedirs("../models", exist_ok=True)
    if (len(os.listdir("../models")) == 0 or force_train) and not cheat:
        process_data("../data/trainData.csv", as_path=True, force=force_process)
        dataloaders = create_dataloaders()
        model = TinyModel(24)
        iterations = 5 # Up to 9 should be fine
        trainTinyModel(dataloaders, model, iterations=iterations)
    else:
        model = load_model()


    model.eval()
    #print("windowDara:",windowData)
    windowData, minimum, maximum= process_window(windowData, normalise= True)
    #print(windowData)
    #print(minimum, maximum)
    threshold = max(windowData['timestamp'].values)-1
    window_X = week_to_X(windowData, threshold, transform=ToTensor())
    #print("Input:")
    #print(window_X)
    #print(window_X.shape)
    #print(window_X.dtype)

    if cheat:
        data =  window_X.tolist()
        #pred = np.array(pred)#.resize(7,24)
        #pred = np.resize(pred, (7,24))
        hours = []
        for hour in range(24):
            hour_data = []
            for day in range(7):
                index = day*24 + hour
                hour_data.append(data[index])
            hours.append(hour_data)
            #print(hour_data)
        #print(hours)
        pred = []
        for hour in hours:
            pred.append(np.mean(hour))
        #print(pred)
        #print(len(pred))
        #pred = [np.mean(pred)] * 24
        #print("")
        #print(pred)
        #print("")
        #quit()
    else:
        pred = model(window_X)
        print("Prediction:")
        print(pred)
        print(pred.shape)


        pred = pred.tolist()
        print(pred)
        scaled_pred = []
        for p in pred:
            scaled_pred.append(p* (maximum-minimum) + minimum)
        print("Rescaled:")
        print(pred)
        for p in model.parameters():
            print(p)
    n += 1
    return pred

# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../scripts/validationData.csv")
    predict(windowData[0:168], force_train = False, force_process=False, cheat = True)