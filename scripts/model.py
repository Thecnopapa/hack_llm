# Essential

# Pytorch-related
import torch
from torchsummary import summary


##### Our model #####


# Class with the model used
class TinyModel(torch.nn.Module):

    def __init__(self, output_dim):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(9, 100)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, output_dim)

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #print("input:", x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        #print("output:", x)
        return x

class customModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(168, 168)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(168, 24)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #print("input:", x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        #print("output:", x)
        return x

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
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


# For testing:
if __name__ == '__main__':

    model = customModel()
    summary(model, (1, 168))
    x = torch.rand(2,168)
    print("input:", x)
    output = model(x)
    print("output:", output)
    print("output.shape:", output.shape)



