## Sources
##### Candlestick charts
https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh 
##### How to build a model for stock prediction
https://keras.io/api/optimizers/ - keras optimizers
https://keras.io/api/losses/ - keras loss functions

##### Data Loading
https://www.alphavantage.co/documentation/ - AlphaVantage API
https://algotrading101.com/learn/alpha-vantage-guide/ - AlphaVantage tutorial

##### Plots
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html

##### Theory
https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f
https://towardsdatascience.com/is-it-possible-to-predict-stock-prices-with-a-neural-network-d750af3de50b
https://towardsdatascience.com/what-happened-when-i-tried-market-prediction-with-machine-learning-4108610b3422
https://www.datacamp.com/community/tutorials/lstm-python-stock-market

## Conclusion
Stock prediction with LSTMs, as many people found previously, was not very accurate. Resulting measurements of the accuracies yielded high MSE and Mean Absolute Percentage Error.  On average, the **mean absolute percentage error (MAPE) tended to be quite high**, and on average the predicted values were at least 10% lower than the actual values seen.  
The model was further complicated with its inability to predict consistently a high and low value, as sometimes the predicted low would be higher than the predicted high. Furthermore, in the real-world stock market, there are many companies, each with different "factors and laws" governing their stock prices, which a model today simply cannot predict.  
Ostensibly, with a 
