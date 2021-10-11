# Score model on train set.
scores = model.evaluate(x_train, s_train, verbose=1)

# Train set scores
print('Train loss:     %f' % np.mean(scores[0 : NUM_OF_LETTERS + 1]))
acc = 1.
for i in range(NUM_OF_LETTERS):
    acc *= scores[NUM_OF_LETTERS + 1 + i]
#    print(scores[NUM_OF_LETTERS + 1 + i])
print('Train accuracy: %.2f' % (acc * 100.))


# Test set scores
scores = model.evaluate(x_test, s_test, verbose = 1)
print('Test loss:     %f' % np.mean(scores[0 : NUM_OF_LETTERS + 1]))
acc = 1.
for i in range(NUM_OF_LETTERS):
    acc *= scores[NUM_OF_LETTERS + 1 + i]
#    print(scores[NUM_OF_LETTERS + 1 + i])
print('Test accuracy: %.2f' % (acc * 100.))


###################            USER TESTING                         #######################
IMG_ROW, IMG_COLS = 60, 160

# get user input to predict if this model works on non-included data
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# evaluate model based on completely new image(s)
def load_image(filename):
	img = load_img(filename, grayscale = True, target_size = (30, 80))
	img = img_to_array(img)

	# reshape + normalize image data
	img = img.reshape(1, 30, 80, 1)
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
      name=fn, length = len(uploaded[fn])))
  full_path = "/content/" + fn
# read/show image
items = os.listdir('/content')



##########            Perform testing                ###########
img = load_image(full_path)
# load model
model = keras.models.load_model('/content/saved_models/keras_trained_model.h5')
# predict the class
digit = model.predict(img)


##########            Extract Prediction                ###########
alphabet_all = list('qwertyuiopasdfghjklzxcvbnm0123456789QWERTYUIOPASDFGHJKLZXCVBNM')
# print(digit)

# Extract the prediction with highest probability from the dataset via argmax function.
result = ""
for i in digit:
  for j in i:
    # print(np.argmax(j))
    result += alphabet_all[np.argmax(j)]
print(result)


###########         Diagnostic, generate own image with ImageCaptcha     ##############
# generate tester image
NUM_OF_LETTERS = 4
captcha_text = create_captcha_text(NUM_OF_LETTERS)
create_image_captcha(captcha_text)
# print(captcha_text)
