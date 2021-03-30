import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Building off our ANN_demo model
fashion_mnist = keras.datasets.fashion_mnist # load in dataset
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data() # split our data into x and y training/test sets

# x: arrays of pixels of each picture
# y: category names

# create list of class names in the y_train data
# its indeces are stored in y_train_full
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# DATA RESHAPING!
# In ANN, we converted our 2D data into 1D arrays
# for CNN, we require 3D data: height, width, channels
x_train_full = x_train_full.reshape((60000, 28, 28, 1))   # reshape and add new dimension, creating 28 x 28 x 1
x_test = x_test.reshape((10000, 28, 28, 1))               # one channel as our images are greyscale

# divide our data as we did for ANN so our values lie between 0 and 1
x_train_n = x_train_full/225.0
x_test_n = x_test/225.0

# split our data
x_valid, x_train = x_train_n[:5000], x_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test_n

# set seed for tensorflow and numpy
np.random.seed(42)
tf.compat.v1.set_random_seed(42)

# OUR ARCHICTURE:
# 1) input layer: 28 x 28 x 1
# filter size 3 x 3
# stride 1
# padding: ignores one pixel from each side

# 2) Conv layer: 26 x 26 x 32
# 26 dimension after removing from each side
# 32 filters

# 3) Pooling layer: 13 x 13 x 32
# using 2 x 2 maximum pooling

# 4) Flatten layer

# 5) Dense layer 1
# 300 neurons
# In this layer, if we had not used pooling, we would have 4-5 million parameters :o
# Pooling layer reduced it to ~1.6 million

# 6) Dense layer 2
# 100 neurons

# 7) Output layer
# 10 neurons for 10 classes

#CREATING OUR MODEL
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides=1, padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

#Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",                          # gradient descent optimization
              metrics=["accuracy"])

# enter our data
model_history = model.fit(x_train, y_train, epochs=30,batch_size= 64,
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
model.save("CNNmodel.h5")