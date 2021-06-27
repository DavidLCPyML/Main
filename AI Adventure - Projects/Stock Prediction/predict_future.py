############                                PREDICT FUTURE VALUES                                  ############

stock_quote = pd.read_csv('STOCK_EXTENDED.csv')
#create new dataframe
new_df = stock_quote.filter(['high'])

# scale last 60 days
last_60_days = new_df[-100:].values
last_60_days_scaled = scaler.transform(last_60_days)
#print(last_60_days_scaled)

#create test set
X_test = []
X_test.append(last_60_days_scaled)
#print(X_test)

#convert to numpy & reshape
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#print(X_test)

# get predicted price
pred_price = model.predict(X_test)
#undo scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
