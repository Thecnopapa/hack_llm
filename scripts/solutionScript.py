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
def predict(windowData, force_train=False, force_process=False, simple=True):
    global n
    print(n, end="\r")

    os.makedirs("../models", exist_ok=True) # Creates model folder if it does not exist
    if (len(os.listdir("../models")) == 0 or force_train) and not simple: # Checks whether there are any tarined models saved
        process_data("../data/trainData.csv", as_path=True, force=force_process) # Process data for training
        dataloaders = create_dataloaders() # Generate dataloaders for the data
        model = TinyModel(24) # Initialise the model
        iterations = 5 # Up to 9 should be fine
        trainTinyModel(dataloaders, model, iterations=iterations) # Train the model
    elif not simple:
        model = load_model() # If there are saved models, load the most trained one


    # Process window data and convert to tensors
    windowData, minimum, maximum= process_window(windowData, normalise= True)
    threshold = max(windowData['timestamp'].values)-1
    window_X = week_to_X(windowData, threshold, transform=ToTensor())

    # Because the model does not work so far this simple prediction is left as palceholder
    # Note: it performs poorly (RMSE 43.8 on validation data)

    if simple:
        data =  windowData
        pred = []
        for hour in data["hour"].unique():
            hour_data = data[data["hour"] == hour]
            hour_data = hour_data["NO2"].values
            hour_average = np.mean(hour_data)
            pred.append(hour_average)
    # Returns the average pollution at each hour of a day

    # If the model were to work we would instead predict the output
    else:
        model.eval() # Stop the model from training
        pred = model(window_X) # Get predictions
        print("Prediction:")
        print(pred)# Here the output unfortunately is a nan tensor
        print(pred.shape)


        pred = pred.tolist() # Tensors to list
        print(pred)
        scaled_pred = []
        for p in pred:
            # Scale it back as the output ranges 0-1
            scaled_pred.append(p* (maximum-minimum) + minimum)
        print("Rescaled:")
        print(pred) # Just to get disappointed again
        for p in model.parameters():
            print(p) # Here we can see the model parameters are nan after training
    n += 1
    return pred


# For testing:
# Run this script to test a window from the validation data
if __name__ == "__main__":
    windowData = pd.read_csv("../scripts/validationData.csv")
    predict(windowData[0:168], force_train = True, force_process=False, simple = True)