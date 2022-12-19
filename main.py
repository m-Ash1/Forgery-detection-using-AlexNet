import os

import cv2
import numpy
import numpy as np
from PIL import Image
import pandas as pd
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

dataSet = []


def createDataSet():
    for img in os.listdir(DATADIR):
        try:
            img = Image.open(os.path.join(DATADIR, img))
            IMG_SIZE = 227
            resizedImage = img.resize((IMG_SIZE, IMG_SIZE))
            data = np.asarray(resizedImage, dtype="int32")
            dataSet.append(data)
        except Exception as e:
            print(str(e))


createDataSet()

# # PLOTTING IMAGE
# for img in dataSet:
#     imgplot = plt.imshow(img)
#     plt.show()


# prepare cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

model = keras.Sequential()

def alexnet():
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
    model.add(Flatten())

    model.compile(loss=keras.losses.binary_crossentropy, optimizer="sgd", metrics=['accuracy'])
    model.summary()

alexnet()

trainingSet = []
testSet = []
trainingLabels = []
testLabels = []
trainingBatches = []
testBatches = []
trainingLabelsBatches = []
testLabelsBatches = []


# Splitting data set
for train, test in kfold.split(dataSet):
    for trainIterator in train:
        trainingSet.append(dataSet[trainIterator])
        trainingLabels.append(1)
    for testIterator in test:
        testSet.append(dataSet[testIterator])
        testLabels.append(1)
    trainingBatches.append(trainingSet)
    testBatches.append(testSet)
    trainingLabelsBatches.append(trainingLabels)
    testLabelsBatches.append(testLabels)

    trainingSet = []
    testSet = []
    trainingLabels = []
    testLabels = []

trainingI = np.array(trainingBatches[0])
labelsI = np.array(trainingLabelsBatches[0])


results = model.fit(trainingI,labelsI, epochs=50)

# for i in trainingI:
#     results = model.fit(i.reshape(1,227,227,3),labelsI)


# , epochs=10, validation_data=None, steps_per_epoch=1, validation_steps=2
#
# for train in trainingSet: # [720,720,720,...]
#     targetData = np.ones_like(train)
#     print(targetData.shape)
#     # # print(batch.shape)
#     # train = train.reshape(227, 227,3)
#     print(train.shape)
#     results = model.fit(train,np.asarray(targetData).astype('float32').reshape((-1, 1)), epochs=50, validation_data=None, steps_per_epoch=7, validation_steps=2)
#
#
# # index = 1
# # for test in testSet:
# #     for batch in test:
# #
