import os

import numpy.ma.extras
import pandas.core.common
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
from dataload import tensors as dataTensors
from dataload import dataloader
from dataload import data
from model import model


print(data[1])
print(dataTensors[1])
tensor = torch.randn(1, 5)
model(tensor)