import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist # load in dataset
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data() # split our data into x and y training/test sets

# x: arrays of pixels of each picture
# y: category names

# create list of class names in the y_train data
# its indeces are stored in y_train_full
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# DATA NORMALIZATION
# normalize picture dimensions so they are approximately the same scale:
# since we know each image's pixel intensity is within range of 0 to 255, we can divide directly
# if unknown: subtract mean from x_train and divide by standard deviation

x_train_n = x_train_full/225.0
x_test_n = x_test/225.0

# further split training data into train/validation:
# Training data = used for training the model
# Validation data = used for tuning hyperparameters and evaluate the models, optimize performance
# Test data = used to test the model

x_valid, x_train = x_train_n[:5000], x_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test_n

# Creating the model architecture
# Two APIs for defining a model in Keras:
# 1) sequential model API
# 2) Functional API

# for our simple example, we use sequential model API

np.random.seed(42)   # used to replicate same result every time, using the same number next time generates same result
tf.compat.v1.set_random_seed(42)

model = keras.models.Sequential()                           # initialize our sequential model
model.add(keras.layers.Flatten(input_shape=[28, 28]))       # Add layer where our 2D matrices are turned into 1D
model.add(keras.layers.Dense(300, activation="relu"))       # Add dense hidden layer with activation ReLU, 300 neurons
model.add(keras.layers.Dense(100, activation="relu"))       # Add dense hidden layer with activation ReLu, 100 neurons
model.add(keras.layers.Dense(10, activation="softmax"))     # Add output layer with activation softmax (since it is classification)

# check our model using summary!
print(model.summary())

# for details of the compile method: https://keras.io/models/sequential/
model.compile(loss="sparse_categorical_crossentropy",   # loss function chosen because y is available in the form of labels
              optimizer="sgd",                          # perform back propagation
              metrics=["accuracy"])

#fit our data!
model_history = model.fit(x_train, y_train, epochs=30,
                    validation_data=(x_valid, y_valid))

# to view our training parameters and history:
print(model_history.params)
print(model_history.history)

# TESTING AND USING OUR MODEL!

print("our model accuracy is: ", model.evaluate(x_test, y_test))

# take 3 inputs from our test data to act as new, unseen data
x_new = x_test[:3]

y_proba = model.predict(x_new)  # outputs probability of each test object being in each class
y_proba.round(2)

y_pred = model.predict_classes(x_new)   # outputs index of the predict class
print(y_pred)
print(np.array(class_names)[y_pred])    # use our class names list to directly print the names of the predicted classes

# save our model!
model.save("model.h5")