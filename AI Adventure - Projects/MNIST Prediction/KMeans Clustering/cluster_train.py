# Importing the dataset from keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.min())
print(x_train.max())
print(image.min())
print(image.max())

# Normalize & reshape x_train
x_train = x_train.astype('float32')
x_train = x_train/255.0
x_train = x_train.reshape(60000,28*28)

# Training the model
kmeans = MiniBatchKMeans(n_clusters=256)
kmeans.fit(x_train)
# extract labels from clusters
reference_labels = retrieve_info(kmeans.labels_,y_train)
number_labels = np.random.rand(len(kmeans.labels_))

for i in range(len(kmeans.labels_)):
  number_labels[i] = reference_labels[kmeans.labels_[i]]
