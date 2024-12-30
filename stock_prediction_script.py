import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Download Nvidia stock data (for example, from Jan 2012 to Jan 2023)
stock_data = yf.download('NVDA', start='2012-01-03', end='2023-01-01')

# Display the first few rows of data
print(stock_data.head())

# Extract the 'Open' price for training
open_prices = stock_data[['Open']]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(open_prices)

# Display the normalized data
print(scaled_data[:5])

# Function to create the dataset structure
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Take the last `time_step` rows as input
        y.append(data[i, 0])  # The target is the next day's value (Open Price)
    return np.array(X), np.array(y)

# Split data into train and test sets
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

# Create the X_train and y_train
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape X_train and X_test to be 3D [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predicting stock prices for the test set
predictions = model.predict(X_test)

# Inverse transform the predictions to get actual stock prices
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Prepare the data for plotting
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, color='blue', label='Actual Nvidia Stock Price')
plt.plot(predictions, color='red', label='Predicted Nvidia Stock Price')
plt.title('Nvidia Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
