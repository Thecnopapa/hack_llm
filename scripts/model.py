# Essential

# Pytorch-related
import torch
from torchsummary import summary
import torch.nn as nn


##### Our models #####


# Class with the models used
class TinyModel(torch.nn.Module):

    def __init__(self, output_dim):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(168, 168*7)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(168*7, 24)

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #print("input:", x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        #x = self.softmax(x)
        #print("output:", x)
        return x

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



# For testing:
if __name__ == '__main__':

    model = Sequence()
    #summary(models, ([168]))
    x = torch.rand(2,168)
    print("input:", x)
    print("dtype:", x)
    output = model(x)
    print("output:", output)
    print("output.shape:", output.shape)



