IMG_ROW, IMG_COLS = 60, 160
num_alphabet = len(alphabet_all)

def load_data(path, test_split = 0.1):
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r = root, d = directories, f = files
    for r, d, f in os.walk(path):
        for fl in f:
            if '.png' in fl:
                flr = fl.split()[0]
                # print(flr[0:NUM_OF_LETTERS])

                # Generates a one-hot encoding of the label.
                # The resulting array will consist of 1s and 0s
                # with a 1 indicating an appearance of a character at a certain position.
                label = np.zeros((NUM_OF_LETTERS, num_alphabet))
                for i in range(NUM_OF_LETTERS):
                    #print(flr[i])
                    label[i, alphabet_all.index(flr[i])] = 1

                img = cv2.imread(os.path.join(r, fl))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, 
                                 (int(IMG_COLS/2), int(IMG_ROW/2)),
                                 interpolation = cv2.INTER_AREA
                                 )
                img = np.reshape(img, 
                                 (img.shape[0], img.shape[1], 1)
                                 )

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print('dataset size:', NUM_SAMPLES, '(train = %d samples, test = %d samples)' % (len(y_train), len(y_test)))
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
  
  # demonstrate that the train-test split procedure is repeatable
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
DATA_PATH = '/content/'

# create sets and normalize
x_train, y_train, x_test, y_test = load_data(DATA_PATH)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# debugging purposes to confirm
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Need to do some further processing on the labels before actually using it. 
# We want to reshape the labels to better fit the one-hot encoding representation.
s_train = []
s_test = []
for i in range(NUM_OF_LETTERS):
    s_train.append(y_train[:, i, :])
    s_test.append(y_test[:, i, :])

# debugging purposes
print(np.shape(s_train))
print(np.shape(s_test))
