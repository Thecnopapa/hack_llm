# Essential
import os
import pandas as pd
import numpy as np

# Pytorch
import torch.utils.data

# Other scripts
from process import process_data
from dataload import PollutionDataset, ToTensor


##### Our solution #####


# To store processed training data
trainData = None
trainTensors = None
trainDataLoader = None

# Our predict function
def predict(windowData, station = "average"):
    global trainData
    global trainTensors
    global trainDataLoader

    if trainDataLoader is None:
        if trainTensors is None:
            if trainData is None:
                # Load or create processed training data (slow) / First-run only & only when necessary
                if not "trainData_processed.csv" in os.listdir("../data"):
                    print("Processed training data not found. Processing training data")
                    trainData, stations = process_data("../data/trainData.csv", name ="trainData", as_path=True)
                else:
                    print("Loading processed training data")
                    trainData = pd.read_csv("../data/trainData_processed.csv")
                print("Training data processed and laoded")
            else:
                print("Processed training data already loaded")

            # Make training tensors
            trainTensors = PollutionDataset(trainData, transform=ToTensor())
        else:
            pass
        trainDataloader = torch.utils.data.DataLoader(trainTensors, batch_size=1, shuffle=False, num_workers=0)
    else:
        pass
    # Process test dataframe (aka windowToPredict)
    print("Processing window data")
    windowData = process_data(windowData, name="windowData")
    print(windowData)

    return



# For testing:
if __name__ == "__main__":
    windowData = pd.read_csv("../validationData.csv")
    predict(windowData)