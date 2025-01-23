# Essential
import os
import pandas as pd

# Pytorch
import torch
import torch.nn as nn

# Other scripts
from utilities import  *


def save_model(model, model_path = "../models/models.pth"):
    print("Saving trained models")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    print("Saved PyTorch Model State {}".format(model_path))

def load_model(model_path="../models/models.pth"):
    models = os.listdir("../models")
    highest_model = (sorted(models))[-1]
    print("Loading models:", highest_model)
    model_path = os.path.join("../models", highest_model)
    model = torch.load(model_path)
    return model



def test(dataloader, model):
    loss_fn = torch.nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            X, y = sample["time"], sample["values"]
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def trainTinyModel(dataloaders, model, iterations=5):
    loss_fn = torch.nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(models.parameters(), lr=1e-3)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)
    for iteration in range(iterations):
        print(f">> Training iteration: {iteration}")
        for dataloader in dataloaders:
            print(f"# Dataloader: {dataloader.name}")
            progress = ProgressBar(len(dataloader))
            for batch, sample in dataloader:
                X, y = sample
                #print(X.shape)
                #print(y.shape)
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                def closure():
                    optimizer.zero_grad()
                    out = model(X)
                    loss = criterion(out, y)
                    #print(out)
                    #print(y)
                    #print('loss:', loss.item())
                    loss.backward()
                    return loss

                optimizer.step(closure)
                progress.add()
        save_model(model,model_path="../models/model_{}.pth".format(iteration))









# For testing:
if __name__ == "__main__":

    from dataload import dataloader as dataloaderTrain
    from model import model

    trainModel(dataloaderTrain, model)