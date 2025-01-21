#We will use pandas to export the data
#After exploring the values we will check for nuls/anomalies

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

data = pd.read_csv('trainData.csv') #Import data
nulls = (data.isnull().sum()) #We check if there are null values

#If we have null values we can impute or delate the row
#Since for the temp models we need complete sequencies we will be imputing the data

while nulls != 0:
    data['NO2'] = data['NO2'].interpolate()
    nulls = (data.isnull().sum())

#We will take a look to ensure that the data is fully in order
#Since temp models like LSTM need the data in the correct order to find sequencial patrons

data = data.sort_values(by=['date','station','hour'])
print(data.head()) #Used for debbuging, made to DELETE later

#This part of the code is used to transform the data in sequences of 168 hours (1 week) and then
#used to predict the next 24h
#window is used to define the input window and prediction for the window output,
#we don't call both veriables window to make it more clear

def createWindow(data, window_size=168, prediction_size=24):
    wind, pred = [], [] #wind as in window and pred as in prediction
    for i in range(len(data) - window_size - prediction_size): #We take off this sizes from the legth of the data to asure we are not trying to access data that doesn't exist
        wind.append(data[i:i+window_size]) #has the input windows for the model
        pred.append(data[i+window_size:i+window_size+prediction_size]) #has the prediction correscponding to each window
    return np.array(wind), np.array(pred)

wind_train, pred_train = createWindow(data['NO2'].values) #Here we use to function with the NO2 data

#Here we will escalate the data to asure the model is not influenced by high magnitudes

scaler = MinMaxScaler()
wind_train = scaler.fit_transform(wind_train.reshape(-1, 1)).reshape(wind_train.shape)
pred_train = scaler.fit_transform(pred_train.reshape(-1, 1)).reshape(pred_train.shape)

joblib.dump(scaler, 'scaler.pk1') #We save the scaler to use it on the validation and test phases

#We're gonna do one last check on the data to ensure that everything is in its place. This is for debbuging, we should DELETE it :P if it's not deleted, hi :)

print("wind_train shape:", wind_train.shape)
print("pred_train shape:", pred_train.shape)
print("NaN a wind_train:", np.isnan(wind_train).sum())
print("NaN a pred_train:", np.isnan(pred_train).sum())

#Whenever we validate the model we will have to use the same preprocessing, here written:

#val_data = pd.read_csv('validationData.csv') 
#scaler = joblib.load('scaler.pk1')

#wind_val, pred_val = createWindow(val_data['NO2'].values)
#wind_val = scaler(wind_val.reshape(-1,1)).reshape(wind_val.shape)
#pred_val = scaler(pred_val.reshape(-1,1)).reshape(pred_val.shape)
