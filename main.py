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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import keras
# from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
# from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import tensorflow_addons as tfa

DATADIR = "D:/Computer Science/Graduation Project/Datasets/CAISA V.1/"

CATEGORIES = ["Au", "Sp"]

dataSet = []


def createDataSet():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                IMG_SIZE = 227
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                dataSet.append([new_array, class_num])

            except Exception as e:
                print(str(e))

createDataSet()


# PLOTTING IMAGE
# imgplot = plt.imshow(dataSet[801][0])
# plt.show()



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

    model.add(layers.Dense(1, activation="softmax"))
    # model.add(Flatten())

    model.compile(loss=keras.losses.binary_crossentropy, optimizer="sgd", metrics=['accuracy'])
    model.summary()

alexnet()

kf = KFold(n_splits=2, shuffle=True,random_state=0)



accuracies = []
# Loop over the k-fold splits
for train_index, test_index in kf.split(dataSet):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # print(test_index)
    # Get the training and testing data
    for i in train_index:
        X_train.append(dataSet[i][0])
        y_train.append(dataSet[i][1])
    for j in test_index:
        X_test.append(dataSet[j][0])
        y_test.append(dataSet[j][1])

    # imgplot = plt.imshow()
    # plt.show()

    # Fit the model on the training data
    model.fit(np.array(X_train), np.array(y_train))
    #
    #
    # Make predictions on the testing data
    y_pred = model.predict(np.array(X_test))

    # Calculate the accuracy score and store it
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    #

avgAccuracy = np.mean(accuracies)
print("Average accuracy:", avgAccuracy)