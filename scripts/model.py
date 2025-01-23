# Essential

# Pytorch-related
import torch



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

        self.linear1 = torch.nn.Linear(168, 1000)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1000, 24)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #print("input:", x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        #print("output:", x)
        return x




# For testing:
if __name__ == '__main__':

    model = customModel()

    x = torch.rand(2,168)
    print("input:", x)
    output = model(x)
    print("output:", output)
    print("output.shape:", output.shape)



