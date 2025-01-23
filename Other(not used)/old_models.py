


import torch
import torch.nn as nn



class customModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(168, 168)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(168, 24)
        self.softmax = torch.nn.Softmax()


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(168, 168*7)
        self.activation = torch.nn.ReLU()
        self.lstm2 = nn.LSTMCell(168*7, 168)
        self.linear = nn.Linear(168, 24)

    def forward2(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 168, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 168, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 168, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 168, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            print(input_t)
            print(input_t.dtype)
            print(h_t, c_t)
            print(h_t.si, c_t.dtype)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, hx, ch):
        print("input:", x)
        h, c = self.lstm1(x, (h,c))
        h2, c2 = self.lstm2(x)
        x = self.linear(x)
        #print("output:", x)
        return x








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