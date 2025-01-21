import os

import torch
import pandas as pd
import numpy as np
from process import *


data_path = "../data/trainData.csv"



def predict(windowToPredict):
    data = format_time(windowToPredict)
    print(data)


    return