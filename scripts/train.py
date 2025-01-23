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
        save_model(model, path="../models/model_{}.pth".format(iteration))





def train_old(dataset, model, loss_fn, optimizer):
    model.train()
    criterion = nn.MSELoss()
    progress = ProgressBar(len(dataloader))

    for batch, sample in enumerate(dataloader):
        X, y = sample["time"], sample["values"]

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        def closure():
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        progress.add()


def trainModel(dataloader, model):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for sample in dataloader:
        timestamp = sample["timestamp"]
        values = sample["values"]
        time = sample["time"]
        print("Timestamp:", timestamp)
        print("Time:", time, type(time))
        print("Values:", values)
        break

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer)

    print("Saving trained models")
    model_path = os.path.join("../models/models.pth")
    state_path = os.path.join("../models/state.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    torch.save(model.state_dict(), state_path)
    print("Saved PyTorch Model State {}".format(model_path))
    return model


def train(dataloaders, model, iterations = 5):
    criterion = nn.MSELoss()


    optimiser = torch.optim.LBFGS(model.parameters(), lr=0.8)

    for iteration in range(iterations):
        print(f">> Training iteration: {iteration}")
        for dataloader in dataloaders:
            print(f"# Dataloader: {dataloader.name}")
            progress = ProgressBar(len(dataloader))
            for batch, sample in dataloader:
                X, y = sample
                print("X shape:", X.shape, X.dtype)
                print("y shape:", y.shape, y.dtype)
                pred = model(X)
                loss = criterion(pred, y)
                # Compute prediction error
                def closure():
                    optimiser.zero_grad()
                    out = model(X)
                    loss = criterion(out, y)
                    print('loss:', loss.item())
                    loss.backward()
                    return loss
                optimiser.step(closure)


def test_sequence(window_X, model):
    with torch.no_grad():
        future = 24
        pred = model(window_X, future=24)
        loss = criterion(pred[:, :-future], 24)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
        return y



# For testing:
if __name__ == "__main__":

    from dataload import dataloader as dataloaderTrain
    from model import model

    trainModel(dataloaderTrain, model)