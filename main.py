import os

import cv2
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

dataSet = []


def createDataSet():
    for img in os.listdir(DATADIR):
        try:
            img_array = Image.open(os.path.join(DATADIR, img))
            IMG_SIZE = 227
            new_array = img_array.resize((IMG_SIZE, IMG_SIZE))
            data = np.asarray(new_array, dtype="int32")
            dataSet.append(data)
        except Exception as e:
            print(str(e))


createDataSet()


# PLOTTING IMAGE
# for img in training_data:
#     imgplot = plt.imshow(img)
#     plt.show()


numpyDataSet = np.array(dataSet)

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
testSet = []
trainingBatches = []
testBatches = []


# Splitting data set
for train, test in kfold.split(numpyDataSet):
    for trainIterator in train:
        trainingSet.append(numpyDataSet[trainIterator])
    for testIterator in test:
        testSet.append(numpyDataSet[testIterator])
    trainingBatches.append(trainingSet)
    testBatches.append(testSet)
    trainingSet = []
    testSet = []


targetData = np.ones(720)
trainingI = trainingBatches[0]

for i in trainingI:
    print(i.shape)
    results = model.fit(i, epochs=50, validation_data=None, steps_per_epoch=7, validation_steps=2)

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
