import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title('stock market trend prediction')
st.divider()
st.sidebar.header('User Input')

STOCK = st.sidebar.text_input("Enter Stock Ticker", 'SBIN.NS')
start_date = st.sidebar.date_input("Enter Start Date", dt.date(2000,1,1))
end_date = st.sidebar.date_input("Enter End Date", dt.datetime.now().date())

df = yf.download(STOCK,start=start_date, end=end_date, progress=False)
st.subheader('Stock " '+STOCK+' ('+str(pd.to_datetime(start_date).date())+' - '+str(pd.to_datetime(end_date).date()))

st.header('Closing Price VS Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.legend(['Closing Price'])
plt.title("Closing Price Vs Time Chart")
plt.grid()
st.pyplot(fig)

st.header("Closing price Vs Time Chart with 100 & 200 Days Moving Average")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.legend(('Closing Price Vs Time Chart with 100 Days Moving Average'))
plt.grid()
st.pyplot(fig)

st.header('Closing Price Vs Time Chart with 100 & 200 Days Moving Average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.legend(['Closing price', '100 Days Moving Average', '200 Days Moving Average'])
plt.grid()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df*0.70))])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler =scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.sidebar('Preficitons Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b')
plt.plot(y_predicted, 'r')
plt.legend(['Original Price', 'Predicted Price'])
plt.title("Predection Vs Original")
plt.grid()
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

