import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow import keras



train_dir = r'dataset/train'
validation_dir = r'dataset/validation'
test_dir =r'dataset/test'

# DATA PROCESSING:
# Read picture files
# Decode JPEG into RBG grids
# Convert into floating point tensors
# Rescale the pixel values


# First run, without augmentation:
# train_datagen = ImageDataGenerator(rescale=1./255)

# With augmentation:
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),     # standardize image size
        batch_size=32,              # feed in batches of 32
        class_mode='binary')        # binary classification

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# create our model!
# generally double # of filters of each conv layer
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))  # deactivate 50% of neurons per epoch, reduces over fitting

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',               # binary cross entropy for 2 classes
              optimizer=optimizers.RMSprop(lr=1e-4),    # RMSprop better for image processing
              metrics=['acc'])

# fit training model
history = model.fit_generator(  # use fit_generator rather than fit as our data was generated above
      train_generator,
      steps_per_epoch=100,      # 32 batches * 100 steps = 3200 images
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50)      # 32 batches * 50 steps = 1600 images

#check our accuracy!
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# model.save("dog_vs_cats_model.h5")
model.save("dog_vs_cats_model_with_augmentation.h5")