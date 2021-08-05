# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

...
# Load dataset
url = "winequality-red-test.csv"
names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
dataset = read_csv(url, names=names)

# correct/clean/normalize the inputted data
dataset = dataset[1:]
dataset = dataset.astype(float)
dataset.quality = dataset.quality.astype(int)

# ensure everything is in order
'''
print(dataset.shape)
print(dataset.dtypes)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('quality').size())
'''

# visualize intercolumnal relationships
'''
dataset.plot(kind = 'box', subplots=True, layout = (4,4), sharex=False,sharey=False, figsize=(10,10))
dataset.hist(figsize=(10,10))
scatter_matrix(dataset, figsize = (20,20))
'''

array = dataset.values
X = array[:,0:11]
y = array[:,11]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# algorithm checking
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name,model in models:
	kfold = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	#print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# using DTree Classifications since it scored the highest
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# look at diagnostics
print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

'''
fig = pyplot.figure(figsize =(10, 10))
pyplot.boxplot(results, labels=names)
pyplot.title('Comparisons')
pyplot.show()
'''
