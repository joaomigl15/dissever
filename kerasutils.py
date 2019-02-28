import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout


def compilecnnmodel(cnnmod):
    if(cnnmod == 'lenet'):
        mod = Sequential()
        mod.add(Conv2D(filters=14, kernel_size=3, padding="same", input_shape=[7, 7, 3], activation='relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Conv2D(filters=28, kernel_size=3, padding="same"))
        mod.add(Activation('relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Dropout(rate=0.1))
        mod.add(Flatten())
        mod.add(Dense(units=56))
        mod.add(Activation('relu'))
        mod.add(Dense(units=1))
        mod.add(Activation('linear'))

    mod.compile(loss='mean_squared_error', optimizer='adam')
    return mod


def createpatches(X, patchsize):
    rowpad = int((patchsize-1)/2)
    colpad = int(round((patchsize-1)/2))

    print(X.shape)

    newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
    patches = extract_patches_2d(newX, (patchsize, patchsize))

    return patches