# Essential
import os
import pandas as pd

# Pytorch
import torch

# Other scripts
from utilities import  *




def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    progress = ProgressBar(len(dataloader))
    for batch, sample in enumerate(dataloader):
        X, y = sample["time"], sample["values"]

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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

    print("Saving trained model")
    model_path = os.path.join("../model/model.pth")
    state_path = os.path.join("../model/state.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    torch.save(model.state_dict(), state_path)
    print("Saved PyTorch Model State {}".format(model_path))
    return model


# For testing:
if __name__ == "__main__":

    from dataload import dataloader as dataloaderTrain
    from model import model

    trainModel(dataloaderTrain, model)