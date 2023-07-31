import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from datetime import date
from datetime import timedelta
import plotly.express as px
import math
today=date.today()
yesterday=today - timedelta(days=1)

import yfinance as yf



st.title('Stock price prediction')
user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download('AAPL',  start='2012-01-01', end=yesterday)



#Describing Data
st.subheader('Data of last 100 days')
last_100_days_df = df.tail(100)

def highlight_row(row):
    '''
    Highlight the maximum in green and minimum in red for each row.
    '''
    color = [''] * len(row)
    if row['High'] == row['High'].max():
        color[row.index.get_loc('High')] = 'background-color: green'
    if row['Low'] == row['Low'].min():
        color[row.index.get_loc('Low')] = 'background-color: red'
    return color

styled_df = last_100_days_df.style.apply(highlight_row, axis=1)
st.write(styled_df)

#visualizations


st.subheader("Closing Price vs Time Chart")

fig = px.line(df,  y="Close", title="Closing Price vs Time",color_discrete_sequence=['mediumturquoise',])
st.plotly_chart(fig, use_container_width=True)

#training dataset
data = df.filter(['Close'])

dataset = data.values # convert to numpy array

training_data_length = math.ceil(len(dataset) * 0.8)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#load my model
model=load_model('keras_model.h5')

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_length - 60:, :]

x_test = []
y_test = dataset[training_data_length: , :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60: i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape again to make it 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #unscaling values


# Get the quote
apple_quote = yf.download('AAPL',  start='2012-01-01', end=yesterday)
new_df = apple_quote.filter(['Close'])

last_60_days = new_df[-60:].values #convert to np array
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_scale_price = model.predict(X_test)

pred_price = scaler.inverse_transform(pred_scale_price)

print(pred_price)
df_pred_price=pd.DataFrame(pred_price,columns=['Predicted Price'])


# Plot the data
import plotly.graph_objects as go

# Plot the data
train = data[:training_data_length]
actual_val = data[training_data_length:]
actual_val['Predictions'] = predictions

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Training Data', line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=actual_val.index, y=actual_val['Close'], mode='lines', name='Actual Value', line_color='green'))
fig.add_trace(go.Scatter(x=actual_val.index, y=actual_val['Predictions'], mode='lines', name='Predicted Value', line_color='orange'))

fig.update_layout(title='Actual value vs Predicted Value', xaxis_title='Date', yaxis_title='Close Price USD($)')
st.plotly_chart(fig)

st.header("predicted stock price for Apple")
st.write(df_pred_price)

ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.Close, mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='MA 100', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='MA 200', line=dict(color='green')))

fig.update_layout(title='Stock Price with Moving Averages for 100 and 200 days', xaxis_title='Date', yaxis_title='Price (USD)')

st.plotly_chart(fig)



