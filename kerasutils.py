import numpy as np
from sklearn.feature_extraction.image import extract_patches
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from itertools import product


def compilecnnmodel(cnnmod, shape, lrate, filters=[2,4,8,16,32], lweights=[1/2, 1/2]):
    if cnnmod == 'cnnlm':
        shape = [7]
        mod = Sequential()
        mod.add(Dense(units=1, input_shape=shape))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'lenet':
        mod = Sequential()
        mod.add(Conv2D(filters=filters[0], kernel_size=3, padding='same', input_shape=shape, activation='relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Conv2D(filters=filters[1], kernel_size=3, padding='same'))
        mod.add(Activation('relu'))
        mod.add(MaxPooling2D(pool_size=(2, 2)))
        mod.add(Dropout(rate=0.1))
        mod.add(Flatten())
        mod.add(Dense(units=filters[2]))
        mod.add(Activation('relu'))
        mod.add(Dense(units=1))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'vgg':
        mod = Sequential()
        mod.add(Conv2D(filters[0], 3, activation='relu', padding='same', input_shape=shape, name='block1_conv1'))
        mod.add(Conv2D(filters[0], 3, activation='relu', padding='same', name='block1_conv2'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
        mod.add(Conv2D(filters[1], 3, activation='relu', padding='same', name='block2_conv1'))
        mod.add(Conv2D(filters[1], 3, activation='relu', padding='same', name='block2_conv2'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv1'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv2'))
        mod.add(Conv2D(filters[2], 3, activation='relu', padding='same', name='block3_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv1'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv2'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block4_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv1'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv2'))
        mod.add(Conv2D(filters[3], 3, activation='relu', padding='same', name='block5_conv3'))
        mod.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
        mod.add(Flatten(name='flatten'))
        mod.add(Dense(filters[4], activation='relu', name='fc1'))
        mod.add(Dense(filters[4], activation='relu', name='fc2'))
        mod.add(Dense(units=1, activation='linear', name='predictions'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'uenc':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        flat1 = Flatten()(drop5)
        dens1 = Dense(units=1, activation='linear', name='predictions')(flat1)
        mod = Model(inputs=inputs, outputs=dens1)
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == 'unet':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(filters[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(filters[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(filters[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(filters[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='linear')(conv9)

        mod = Model(inputs=inputs, outputs=conv10)
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=lrate))

    elif cnnmod == '2runet':
        inputs = Input(shape)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(filters[4], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(filters[3], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(filters[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(filters[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(filters[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(filters[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(filters[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(filters[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)


        # High resolution output
        outputhr = Conv2D(1, 1, activation='linear', name="highres")(conv9)

        # Sum of high resolution output
        avgpoolinghr = AveragePooling2D(pool_size=4)(outputhr)
        outputlr = Lambda(lambda x: x * 4, name="lowres")(avgpoolinghr)

        mod = Model(inputs=inputs, outputs=[outputhr, outputlr])
        mod.compile(loss=['mean_squared_error', 'mean_squared_error'],
                    loss_weights=lweights,
                    optimizer=optimizers.Adam(lr=lrate))

    return mod



def createpatches(X, patchsize, padding, stride=1, cstudy=None):
    if cstudy:
        try:
            fp = np.memmap(cstudy + '.dat', mode='r')
            print('Found .dat file')
            ninstances = int(fp.shape[0] / patchsize / patchsize / X.shape[2] / 4) # Divide by dimensions
            shapemmap = (ninstances, patchsize, patchsize, X.shape[2])
            fp = np.memmap(cstudy + '.dat', dtype='float32', mode='r', shape=shapemmap)
        except:
            print('Did not find .dat file')
            if padding:
                rowpad = int((patchsize - 1) / 2)
                colpad = int(round((patchsize - 1) / 2))
                newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
            else:
                newX = X
            patches = extract_patches(newX, (patchsize, patchsize, X.shape[2]), extraction_step=stride)
            fp = np.memmap(cstudy + '.dat', dtype='float32', mode='w+', shape=patches.shape)
            fp[:] = patches[:]
            fp = fp.reshape(-1, patchsize, patchsize, X.shape[2])
        return fp
    else:
        if padding:
            rowpad = int((patchsize - 1) / 2)
            colpad = int(round((patchsize - 1) / 2))
            newX = np.pad(X, ((rowpad, colpad), (rowpad, colpad), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            newX = X

        patches = extract_patches(newX, (patchsize, patchsize, X.shape[2]), extraction_step=stride)
        patches = patches.reshape(-1, patchsize, patchsize, X.shape[2])
        return patches


def reconstructpatches(patches, image_size, stride):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    n_h = int((i_h - p_h) / stride + 1)
    n_w = int((i_w - p_w) / stride + 1)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
        patch_count[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += 1
    return img/patch_count
