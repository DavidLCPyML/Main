from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from collections import Counter

# load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:,0:8]
Y = dataset[:,8]
# split into train test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=3, stratify=Y)
# NOTE! Stratify only works for classification problems!
print(X_train[:10, :])

# split again, and we should see the same split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
# print(X_train[:10, :])
print(Counter(Y))
print(Counter(Y_train))
print(Counter(Y_test))

# define keras model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
# syntax:  Dense(# nodes, input dimensions, and activation funtion)
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# the most confusing thing here is that the shape of the input to the model is
# defined as an argument on the first hidden layer.
# This means that the line of code that adds the first Dense layer is doing 2
# things, defining the input or visible layer and the first hidden layer.


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
# work happens here, either on CPU or GPU.
model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)


# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
# evaluate returns 2 values, the loss and the accuracy, respectively.
# here, accuracy is favored, so we use that.
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict_classes(X)
# summarize the first 10 cases
for i in range(10):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], Y_test[i]))
