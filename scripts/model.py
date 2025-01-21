

import torch
from dataload import tensors as dataTensors


print(dataTensors)

'''lin = torch.nn.Linear(5, 15)
print(dataTensors[1233]["values"].shape)
x = torch.rand(5,1)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
'''

print("########################################################")
import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(5, 100)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 1)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x



model = TinyModel()

print('The model:')
print(model)

print('\n\nModel params:')
for param in model.parameters():
    print(param)

