


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics, tree, cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RandomizedLogisticRegression
from mnist import MNIST
import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
# NOTE: Make sure that the class is labeled 'class' in the data file

import matplotlib.pyplot as plt
# fix random seed for reproducibility

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())
    # Remember to center and normalize the data...
    return X_train, labels_train, X_test
X_train, label_train, X_test = load_dataset()

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(label_train)
encoded_Y = encoder.transform(label_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # compute mean, std and transform training data as well
X_test = scaler.transform(X_test)
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_dim=784, init='normal', activation='relu',W_constraint = maxnorm(2)))
	model.add(Dense(10, init='normal', activation='sigmoid',W_constraint = maxnorm(2)))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = Sequential()
model.add(Dropout(0.2, input_shape=(784,)))
model.add(Dense(200, input_dim=784, init='normal', activation='relu',W_constraint = maxnorm(2)))
model.add(Dropout(0.2))
model.add(Dense(10, init='normal', activation='sigmoid',W_constraint = maxnorm(2)))
# Compile model
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
#estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=2, batch_size=50, verbose=0)
model.fit(X_train,dummy_y, nb_epoch=2, batch_size=50)
#predictions = estimator.predict(X_test)
'''
history = model.fit(X_train, dummy_y, validation_split=0.167, nb_epoch=15, batch_size=50, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
import pydot_ng
pydot_ng.find_graphviz()