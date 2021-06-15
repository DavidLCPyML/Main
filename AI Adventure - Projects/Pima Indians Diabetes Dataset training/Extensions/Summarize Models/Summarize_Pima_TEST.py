# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
from keras.utils.vis_utils import plot_model

# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
