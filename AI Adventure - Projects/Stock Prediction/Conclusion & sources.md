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
Stock prediction with LSTMs, as many people found previously, was not very accurate. Resulting measurements of the accuracies yielded high error values.  On average, the **mean absolute percentage error (MAPE) tended to be quite high**, and on average the predicted values were at least 10% lower than the actual values seen. The model was further complicated with its **inability to consistently and correctly predict a high and low value**, as sometimes **the predicted low would be higher** than the predicted high. 
This **did not only apply to LSTMs, however**. Different models (such as ARIMA and Decision Tree regressors) each **failed to properly predict the high and low prices**, falling victim to the same types of errors the LSTM encountered, such as underpredicting high values and overpredicting low values. For the models investigated, the predictions seemed **no better than a datashift to the left one day**.
Furthermore, in the real-world stock market, there are many companies, each with **different "factors and laws" governing their stock prices**, which a model today simply cannot predict. One could theoretically train everything towards one company, but doing that with every major company would not only **require insane amounts of computing power**, but also **intimate insider knowledge of said "laws" and "factors"**.
Perhaps models like these could prove useful in microtrading circumstances, but for long-term or even daily prediction linear regression and human intuition does far better. This is because **the network tries to guess future values based on past values**. In real-time trading, if the price grows for some time, **the computer would still predict infinitely increasing returns until a fall, but a more experienced trader might see negative news and anticipate a fall**. Stock price is **affected only by external events** the computer cannot predict ahead of time.
Machine Learning, at least for stocks, has a long way to go before being a true asset to market prediction.
##### speculation based on my findings
One might argue that technical analysis combined with news combing would lead to a decent predictor. **That may be true**, and is perhaps what most companies use as a tool to gauge opinion on investing or selling. However, adding extra models (such as NLP to interpret news opinions) to one that already takes a while to simulate cripples a model even further. In addition to higher time cost, it would also be difficult to account for every possible "factor and law" governing the stock, just like the previous technical analysis models.
This is further complicated with the somewhat paradoxical idea that if such a model were to exist, so many people would begin using it to the point where the model cannot predict prices as people everywhere begin relying on it for "free money".
