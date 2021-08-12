import matplotlib
#matplotlib.use('Agg')
import numpy as np
import sys
import sklearn
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
print(type(x_train), ' ', type(y_train))
print(type(x_test), ' ', type(y_test))

# visualize dataset
fig, axs = plt.subplots(3, 3, figsize = (12, 12))
plt.gray()
for i, ax in enumerate(axs.flat):
  ax.matshow(x_train[i])
  ax.axis('off')

  display_width = y_train[i]
  text = y_train[i]
  # print(f'Number {text}')
  ax.set_title(f'Number {text}')


# Preprocessing
# print(x_train.max())
# print(x_train.min())
x_train = x_train.astype('float32')
x_test  = x_test .astype('float32')
x_train = x_train / 255.0
x_test  = x_test  / 255.0
# print(x_train.max())
# print(x_train.min())
X_train = x_train.reshape(len(x_train), -1)
X_test  = x_test .reshape(len(x_test ), -1)
print(X_train.shape)
print(X_test .shape)
