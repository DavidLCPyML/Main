from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:,0:8]
Y = dataset[:,8]

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
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)


# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
# evaluate returns 2 values, the loss and the accuracy, respectively.
# here, accuracy is favored, so we use that.
print('Accuracy: %.2f' % (accuracy*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
