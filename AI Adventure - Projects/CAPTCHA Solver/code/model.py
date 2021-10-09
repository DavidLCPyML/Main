input_layer = Input((int(IMG_ROW/2), int(IMG_COLS/2), 1))
# convolutional filters
x = Conv2D(filters = 32, kernel_size = (5, 5), padding = 'same', activation = 'relu')(input_layer)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
#connected layers
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.3)(x)
# output layer
out = [Dense(num_alphabet, name = 'digit%d' % i, activation = 'softmax')(x) for i in range(NUM_OF_LETTERS)]
model = Model(inputs = input_layer, outputs = out)


# initialize with Adam optimizer, use binary crossentropy(ideal for multiclass prediction)
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
model.summary()
