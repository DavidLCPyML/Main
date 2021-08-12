from google.colab import files
import os
import cv2

from skimage import color
from skimage import io
from skimage.transform import resize

# upload files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# read/show image
items = os.listdir('/content')
# print(items)    
for each_image in items:
  if each_image.endswith(".png"):
    # print(each_image)
    full_path = "/content/" + each_image
    print(full_path)
    image = plt.imread(full_path)
'''
plt.imshow(image)
plt.colorbar()
plt.grid(False)
image.shape
'''

# process image/convert RGB to monochrome
image = plt.imread(full_path)
image = resize(image, (28, 28))
image = color.rgb2gray(image)

#'''
plt.imshow(image)
plt.colorbar()
plt.grid(False)
image.shape
#'''

# Reshaping into a row vector
image = image.reshape(1, 28*28)
# image = image * 1/image.max()
# predict the clusters that each image goes to
predicted_cluster = kmeans.predict(X_test)
print(number_labels[predicted_cluster])

print(f'Accuracy: {accuracy_score(number_labels,y_train)}')
