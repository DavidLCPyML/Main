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
