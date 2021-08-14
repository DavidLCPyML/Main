# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = prep_pixels(trainX, testX)
	
  # evaluate model
	model = define_model()
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	model.save('final_model.h5')

run_test_harness()

from keras.models import load_model

trainX, trainY, testX, testY = load_dataset()
trainX, testX = prep_pixels(trainX, testX)

# load model
model = load_model('final_model.h5')
_, acc = model.evaluate(testX, testY, verbose=0)
print(acc * 100.0)
