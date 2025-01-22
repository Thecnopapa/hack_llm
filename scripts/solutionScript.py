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
from model import TinyModel
from train import trainModel, test


##### Our solution #####


# To store processed training data
trainData = None
trainTensors = None
trainDataLoader = None
model = None

# Our predict function
def predict(windowData, station = "average"):
    global trainData
    global trainTensors
    global trainDataLoader
    global model
    # Load or create all required components if not already done so / First-run only & only when necessary
    if model is None:
        if trainDataLoader is None:
            if trainTensors is None:
                if trainData is None:
                    if not "trainData_processed.csv" in os.listdir("../data"):
                        # Process data if not already processed (Slow!)
                        print("Processed training data not found. Processing training data")
                        trainData, stations = process_data("../data/trainData.csv", name ="trainData", as_path=True)
                    else:
                        # Load processed data if found
                        print("Loading processed training data")
                        trainData = pd.read_csv("../data/trainData_processed.csv")
                    print("Training data processed and loaded")
                else:
                    print("Processed training data already loaded")

                # Make tensors from training data when necessary
                trainTensors = PollutionDataset(trainData, transform=ToTensor())
            else:
                print("Tensors already loaded")

            # Create dataloader from tensors when necessary
            trainDataLoader = DataLoader(trainTensors, batch_size=1, shuffle=False, num_workers=0)
        else:
            print("Dataloader already loaded")

        if not os.path.exists("../model/model.pth"):
            print("Model file not found. Training new model")
            model = TinyModel()
            model = trainModel(trainDataLoader, model)
        else:
            print("Trained model loaded from file")
            model = torch.load("../model/model.pth")
    else:
        print("Trained model already loaded")
    print("Model ready")
    print(model)

    # Process test dataframe (aka windowToPredict)
    print("Processing window data")
    windowData, stations = process_data(windowData, name="windowData")
    print(windowData)
    print("Creating window tensors")
    windowTensors = PollutionDataset(windowData, transform=ToTensor(), is_test=True)
    print(windowTensors)
    print("Creating window dataloader")
    windowDataLoader = DataLoader(windowTensors, batch_size=1, shuffle=False, num_workers=0)
    print(windowDataLoader)

    test(windowDataLoader, model)

    return



# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../validationData.csv")
    predict(windowData)