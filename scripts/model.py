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



