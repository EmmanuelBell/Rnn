#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:18:32 2020

@author: emmanuelmoudoute-bell
"""


# Recurrent Neural Networks


# Partie 1 - Préparation des données

# Librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Jeu d'entrainement 
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values


# Feature scaling ( mise à l'échelle)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Création de la structure avec 60 timesteps et 1 sortie
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Partie 2 - Construction  du RNN

# Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialisation
regressor = Sequential()

# Couche LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True, 
                   input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2e couche LSTM +Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 3e couche LSTM +Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 4e couche LSTM +Dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Couche de sortie
regressor.add(Dense(units=1))

# Compliation
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Entrainement 
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Partie 3 - Prédiction et visualisation

# Données de 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Prédictions pour 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()