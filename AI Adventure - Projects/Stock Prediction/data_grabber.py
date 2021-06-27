import os
import requests
import pandas as pd

############                                        PARAMS                                       ############
api_key = 'HXFFE4IZUVLA5GEU'
#print(api_key)
symbol = 'AAPL'
function = 'TIME_SERIES_INTRADAY_EXTENDED'
adjusted = 'true'






############                             GET DATA FROM ALPHAVANTAGE                              ############
base_url = 'https://www.alphavantage.co/query?'
params = {'function': function,
		 'symbol' : symbol, 
		 'interval': '60min',
     'slice': 'year1month3',
     'adjusted': adjusted,
		 'apikey': api_key}
response = requests.get(base_url, params=params)
with open('STOCK_extended3.csv', 'wb') as file:
	file.write(response.content)
#Create pandas dataframe
df3 = pd.read_csv('STOCK_extended3.csv') 


params = {'function': function,
		 'symbol' : symbol, 
		 'interval': '60min',
     'slice': 'year1month2',
     'adjusted': adjusted,
		 'apikey': api_key}
response = requests.get(base_url, params=params)
with open('STOCK_extended2.csv', 'wb') as file:
	file.write(response.content)
#Create pandas dataframe
df2 = pd.read_csv('STOCK_extended2.csv') 


params = {'function': function,
		 'symbol' : symbol, 
		 'interval': '60min',
     'slice': 'year1month1',
     'adjusted': adjusted,
		 'apikey': api_key}
response = requests.get(base_url, params=params)
with open('STOCK_extended1.csv', 'wb') as file:
	file.write(response.content)
#Create pandas dataframe
df1 = pd.read_csv('STOCK_extended1.csv') 


df1 = df1.append(df2)
out = df1.append(df3)
# print(out)
#Time-series index
out.set_index('time', inplace=True)
out.to_csv('STOCK_EXTENDED.csv')







############                             CREATE GRAPHS                              ############
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import mplfinance as fplt

df = pd.read_csv('STOCK_EXTENDED.csv', index_col=0, parse_dates=True)
#dt_range = pd.date_range(start="2021-06-01 04:00:00", end="2020-06-23 20:00:00")
#df = df[df.index.isin(dt_range)]
df.head()

fplt.plot(
            df[:24],
            type='candle',
            style='charles',
            title='Apple, March 2021 - present',
            ylabel='Price ($)',
            volume=True,
            #figratio = (72,48),
            ylabel_lower='Shares\nTraded',
            #show_nontrading=True
            )

#df = pd.read_csv('STOCK_EXTENDED.csv')
#fig = go.Figure(data=[go.Candlestick(x=df['time'],
#                open=df['open'],
#                high=df['high'],
#                low=df['low'],
#                close=df['close'])])
#fig.update_layout(xaxis_rangeslider_visible=False)
#fig.show()


############                             CREATE DATASET                              ############
def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y


############                                DATA PROCESSING                                  ############
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# obtain train off high values
df = pd.read_csv('STOCK_EXTENDED.csv')
df.head()
df = df['high'].values
df = df.reshape(-1, 1)
#print(df.shape)
#df[:5]

#split into train-test datasets
WINDOW = 50
train_test_split = 0.8
dataset_train = np.array(df[:int(df.shape[0]*train_test_split)])
dataset_test = np.array(df[int(df.shape[0]*train_test_split)-WINDOW:])

# scale data to fit btwn range 0 and 1, then create input/output train-test sets
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)
#print(dataset_train.shape)
#print(dataset_test.shape)
#dataset_train[:5]
#dataset_test[:5]
#x_train[:1]
#x_test[:1]

# Reshape features for LSTM Layer
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
