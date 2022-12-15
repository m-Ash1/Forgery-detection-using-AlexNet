import os

import numpy
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from numpy import asarray
from numpy.ma import indices
from sklearn.model_selection import KFold
import keras
# from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
# from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import tensorflow_addons as tfa


DATADIR = "D:\Computer Science\Graduation Project\Datasets\CAISA V.1\Au"  # put your directory here

training_data = []


def create_training_data():
    for img in os.listdir(DATADIR):
        try:
            img_array = Image.open(os.path.join(DATADIR, img))
            IMG_SIZE = 227
            new_array = img_array.resize((IMG_SIZE, IMG_SIZE))
            data = np.asarray(new_array, dtype="int32")
            training_data.append(data)
        except Exception as e:
            print(str(e))


create_training_data()


# PLOTTING IMAGE
# for img in training_data:
#     imgplot = plt.imshow(img)
#     plt.show()


out_images = np.array(training_data)

# prepare cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

model = keras.Sequential()

def AlexNet():
    # C1
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                            strides=(4, 4),
                            input_shape=(227, 227, 3)))
    model.add(tfa.layers.Maxout(96))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    # C2
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding="same"))
    model.add(tfa.layers.Maxout(256))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    # C3
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same"))
    model.add(tfa.layers.Maxout(384))

    # C4
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same"))
    model.add(tfa.layers.Maxout(384))

    # C5
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same"))
    model.add(tfa.layers.Maxout(256))

    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.Flatten())

    model.add(layers.Dense(4096))
    model.add(tfa.layers.Maxout(4096))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096))
    model.add(tfa.layers.Maxout(4096))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation="softmax"))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer="sgd", metrics=['accuracy'])
    model.summary()

AlexNet()

trainingSet = []
TestSet = []

for train, test in kfold.split(out_images):
    # results = model.fit(out_images[train], epochs=50, validation_data=None, steps_per_epoch=7, validation_steps=2)
    trainingSet.append(out_images[train])
    TestSet.append(out_images[test])



print('train: %s, test: %s' % (trainingSet, TestSet))
