import tensorflow as tf
from tensorflow import keras
from keras import layers
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import randint, uniform


# Load data
sp500 = yf.Ticker("BTC-USD")
sp500 = sp500.history(period="max")

# Refine data
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

num_features = sp500.shape[1]
column_names = sp500.columns

# Num of features = 7

# Define the architecture of the neural network
def create_model():
    model = keras.Sequential([
        layers.Dense(units=33, activation='tanh', input_shape=(6,)),
        layers.Dense(units=33, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assuming X is your input features and y is your target variable
X = sp500[["Open", "High", "Low", "Close", "Volume", "Tomorrow"]].values
y = sp500["Target"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create the model
model = create_model()

# Train the model with the specified hyperparameters
model.fit(X_train, y_train, batch_size=32, epochs=20)

# Evaluate the best model using test data
loss, accuracy = model.evaluate(X_test, y_test)


print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
X_new_data = sp500[["Open", "High", "Low", "Close", "Volume", "Tomorrow"]].values
predictions = model.predict(X_new_data)
print("Predictions:", predictions)