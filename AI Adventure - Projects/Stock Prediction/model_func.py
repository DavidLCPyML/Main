############                                DATA PROCESSING                                  ############
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# obtain train off high values
stock_listing = pd.read_csv('STOCK_EXTENDED.csv')
#stock_listing.head()
stock_listing = stock_listing['high'].values
stock_listing = stock_listing.reshape(-1, 1)
#print(stock_listing.shape)
#stock_listing[:5]

#split into train-test datasets
WINDOW = 50
train_test_split = 0.8
train_set = np.array(stock_listing[:int(stock_listing.shape[0]*train_test_split)])
test_set = np.array(stock_listing[int(stock_listing.shape[0]*train_test_split)-WINDOW:])

# scale data to fit btwn 0 and 1, then create input/output train-test sets
scaler = MinMaxScaler(feature_range=(0,1))
train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)


# set up train and test sets
x_train, x_test = [], []
y_train, y_test = [], []
#add elements
for i in range(WINDOW, train_set.shape[0]):
  x_train.append(train_set[i-WINDOW:i, 0])
  y_train.append(train_set[i, 0])
for i in range(WINDOW, test_set.shape[0]):
  x_test.append(test_set[i-WINDOW:i, 0])
  y_test.append(test_set[i, 0])
#convert to numpy
x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

'''
print(train_set.shape)
print(test_set.shape)
print(train_set[:5])
print(test_set[:5])
print(x_train[:1])
print(x_test[:1])
'''

# Reshape features for LSTM Layer
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))




############                                MODEL TRAINING                                  ############
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(25))
model.add(Dropout(0.3))
model.add(Dense(1))

# "bad" optimizers for stocks: SGD (pretty bad), RMSProp (a pretty terrible data shift), ftrl (just a flat line)
# "mediocre" ones: adadelta (smooth graph but misses a lot), adagrad (overpredicts), 
# "decent" ones: adamax (basically shifted 1 over), nadam (same as adamax), adam (probably the best here)
model.compile(loss='huber_loss', optimizer='adam')
if not os.path.exists('stock_model.h5'):
    model.fit(x_train, y_train, epochs=50, batch_size=20)
    model.save('stock_model.h5')
model = load_model('stock_model.h5')





############                                MODEL TESTING                                  ############
#test price predictions in test sets based on train
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print(predictions)


fig, ax = plt.subplots(figsize=(8,4))
plt.plot(stock_listing, label="Actual Price")
ax.plot(range(len(y_train)+WINDOW,len(y_train)+WINDOW+len(predictions)),predictions, color='red', label='Predicted Price')
plt.legend()


#plot results & predictions
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(y_test_unscaled, label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.legend()
