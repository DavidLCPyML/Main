from keras.layers import BatchNormalization
# load data in keras
def load_dataset():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  trainX = x_train.reshape((x_train.shape[0], 28, 28, 1))
  testX  = x_test .reshape((x_test .shape[0], 28, 28, 1))

  # since there are 10 different classes (0-9), one-hot encoding will do
  trainY = to_categorical(y_train)
  testY  = to_categorical(y_test )
  return trainX, trainY, testX, testY

# needs to normalize data to fit [0,1] range (since we use NN)
def normalize(train, test):
  train_norm = train.astype('float32')
  test_norm  = test .astype('float32')
  
  # data is on RGB scale, divide by 255(max possible RGB value)
  train_norm /= 255.0
  test_norm  /= 255.0
  return train_norm, test_norm

'''
Model needs to do two things: 
    1. extract features from the front end, done w/ convolutional and pooling layers, and 
    2. classify images and make a prediction on the backend.

For 1), a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer will do.
  small filter size & low filter # to ensure there are enough distinguishing features
  pooling to add invariance to the noticed features. Batch Normalization for speeding up learning.
  Then flatten to highlight features to the classifier.

For 2):
  Since the numbers are 0-9, we have 10 nodes, so softmax is needed rather than sigmoid.
  Between the feature extractor and the output layer, we can add a dense layer to interpret the features, in this case with 100 nodes.

ReLU activation function and the He_weight initialization scheme are both 'best' choices for this.
Learning rate of 0.01 and a momentum of 0.9 - don't want to learn too fast and end up rushing. 
Categorical cross-entropy as the loss function - for multi-class classification and accuracy as metric since the distribution is approximately uniform.
'''
def create_model():
  model = Sequential()
  model.add(Conv2D            (32, (3,3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (28, 28, 1)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D      ((2,2)))
  model.add(Flatten           ())
  model.add(Dense             (100, activation = 'relu', kernel_initializer = 'he_uniform'))
  model.add(BatchNormalization())
  model.add(Dense             (10,  activation = 'softmax'))
  
  optimize = SGD(lr = 0.01, momentum = 0.9)
  model.compile(optimizer = optimize, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
  return model

# perform kfold cross-validation (k=5, arbitrarily chosen to not be inaccurate or slow)
'''
The training dataset is shuffled prior to being split, performed each time = same train and test datasets in each fold, so easy comparison between models.

Train baseline model for 10 epochs, batch size 32. Test done on each fold during each epoch of the training run and at the end of the run
  during epoch -> create learning curves, end of run -> estimate the performance of the model. 
later graph history and classification accuracy of the fold from each run.
'''
def evaluate(data_x, data_y, n_folds):
  scores, histories = list(), list()
  # cross-validation: Each test set will be 20% of the training dataset, or about 12,000 examples.
  kfold = KFold(n_folds, shuffle = True, random_state = 1)
  # now begin training
  for train_idx, test_idx in kfold.split(data_x):
    model = create_model()
    trainx,trainy,testx,testy = data_x[train_idx], data_y[train_idx], data_x[test_idx], data_y[test_idx]

    # fit + evaluate model
    history = model.fit(trainx, trainy, epochs = 10, batch_size = 32, validation_data = (testx, testy), verbose = 0)
    _, acc = model.evaluate(testx, testy, verbose = 0)
    print(acc)

    # add diagnostics to set
    scores    .append(acc    )
    histories .append(history)
  return scores, histories

# run the test harness for evaluating a model
trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
trainX, testX = normalize(trainX, testX)
scores, histories = evaluate(trainX, trainY, 5)
