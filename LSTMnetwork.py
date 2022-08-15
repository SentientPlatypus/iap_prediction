import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np


dataframe:pd.DataFrame = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
plt.plot(dataframe)
plt.show()

tf.random.set_seed(7)

dataset:np.ndarray = dataframe.values
lengthOfPredictions = len(dataset) * 0.5
dataset = np.append(dataset, np.repeat(np.NaN, lengthOfPredictions))
dataset = dataset.astype('float32')
print(dataset)

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)