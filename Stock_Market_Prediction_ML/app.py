import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model('D:\\Stock Price Prediction\\Stock Predictions Model.keras')

# Streamlit UI setup
st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Error handling for empty data
if data.empty:
    st.error("No data found! Try a different stock symbol or modify the date range.")
    st.stop()  # Stop execution if data is empty

st.subheader('Stock Data')
st.write(data)

# Splitting data for training & testing
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Apply MinMax Scaling
scaler = MinMaxScaler(feature_range=(0,1))

# Handle empty training data scenario
if data_train.empty or data_test.empty:
    st.error("Insufficient data for training. Try adjusting the date range.")
    st.stop()

# Prepare data for prediction
pas_100_days = data_train.tail(100)

# Error handling for empty past days data
if pas_100_days.empty:
    st.error("Not enough past data to generate predictions. Consider selecting a stock with longer trading history.")
    st.stop()

data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving averages
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

# Preparing input for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predictions using the trained model
predict = model.predict(x)

# Reverse scaling
scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# Visualizing predictions
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)