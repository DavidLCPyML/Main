import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
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
