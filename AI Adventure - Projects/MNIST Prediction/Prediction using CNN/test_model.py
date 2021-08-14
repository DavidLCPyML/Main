from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# evaluate model based on completely new image(s)
def load_image(filename):
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	img = img_to_array(img)

	# reshape + normalize image data
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	return img



########          Grab input image from user           #########
from google.colab import files
import os

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
    full_path = "/content/" + each_image
    print(full_path)



##########            Perform testing                ###########
img = load_image(full_path)
# load model
model = load_model('final_model.h5')
# predict the class
digit = model.predict_classes(img)
print(digit[0])
