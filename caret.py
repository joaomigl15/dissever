import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import skimage.measure
import nputils as npu
import kerasutils as ku
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor


def fitlm(X, y):
    mod = LinearRegression()
    mod = mod.fit(X, y)
    return mod

def fitsgdregressor(X, y):
    mod = SGDRegressor(max_iter=100, alpha=0, learning_rate='constant', eta0=0.0001, verbose=1)
    mod = mod.fit(X, y)
    return mod

def fitlmkeras(X, y, batchsize=256, learningrate=0.0001, epoch=100):
    def base_model():
        mod = Sequential()
        mod.add(Dense(units=1))
        mod.add(Activation('linear'))
        mod.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learningrate))
        return mod

    print('Keras LM (wrapper) »»» Adam, BS =', batchsize)
    mod = KerasRegressor(build_fn=base_model, verbose=1, epochs=100, batch_size=batchsize)
    mod.fit(X, y)
    return mod

def fitrf(X, y):
    mod = RandomForestRegressor()
    mod = mod.fit(X, y)
    return mod

def fitxgbtree(X, y):
    mod = xgb.XGBRegressor()
    mod = mod.fit(X, y)
    return mod


def fit(X, y, p, method, batchsize, lrate, epoch):
    X = X.reshape((-1, X.shape[2]))
    y = np.ravel(y)

    relevantids = np.where(y > 0)[0]
    relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
    idsamples = relevantids[relevsamples]
    print('| --- Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

    X = X[idsamples,:]
    y = y[idsamples]

    if(method.endswith('lm')):
        print('|| Fit: Linear Model')
        return fitlm(X, y)
    if (method.endswith('sgdregressor')):
        print('|| Fit: SGD Regressor')
        return fitsgdregressor(X, y)
    if (method.endswith('lmkeras')):
        print('|| Fit: Linear Model - Keras')
        return fitlmkeras(X, y, batchsize=batchsize, learningrate=lrate, epoch=epoch)
    elif(method.endswith('rf')):
        print('|| Fit: Random Forests')
        return fitrf(X, y)
    elif(method.endswith('xgbtree')):
        print('|| Fit: XGBTree')
        return fitxgbtree(X, y)
    else:
        return None


def get_callbacks():
    return [
         ReduceLROnPlateau(monitor='loss', min_delta=0.01, patience=3, factor=0.1, min_lr=5e-6, verbose=1),
         EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=1, restore_best_weights=True)
    ]


def fitcnn(X, y, p, cnnmod, cnnobj, epochs, batchsize, extdataset):
    if cnnmod == 'cnnlm':
        print('| --- Fit - 1 resolution Linear Model')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]
        X = X[:,middrow,middcol]

        relevantids = np.where(y > 0)[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
        else:
            newX = X[idsamples, :]
            newy = y[idsamples]

        return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        # return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize, callbacks=get_callbacks())

    elif cnnmod == 'lenet':
        print('| --- Fit - 1 resolution Le-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(y > 0)[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod == 'vgg':
        print('| --- Fit - 1 resolution VGG-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(y > 0)[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod.endswith('unet'):
        # Compute midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))

        # Train only with patches having middpixel > 0
        relevantids = np.where(y[:,middrow,middcol,0] > 0)[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if(cnnmod == 'unet'):
            print('| --- Fit - 1 resolution U-Net Model')
            if extdataset:
                newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            else:
                newX = X[idsamples, :, :, :]
                newy = y[idsamples]

        elif(cnnmod.startswith('2r')):
            print('| --- Fit - 2 resolution U-Net Model')
            if extdataset:
                newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            else:
                newX, newy = X[idsamples, :, :, :], y[idsamples]
            newylr = skimage.measure.block_reduce(newy, (1,4,4,1), np.sum) # Compute sum in 4x4 blocks
            newy = [newy, newylr]

        return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        # return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize, callbacks=get_callbacks())

    else:
        print('Fit CNN - Unknown model')


def predict(mod, X):
    newX = X.reshape((-1, X.shape[2]))
    pred = mod.predict(newX)
    pred = pred.reshape(X.shape[0], X.shape[1])
    return pred

def predictcnn(obj, mod, ancpatches, disspatcheso, dissshape, batchsize, stride=1):
    if mod == 'cnnlm':
        print('| --- Predicting new values, Keras Linear Model')
        middrow, middcol = int((ancpatches.shape[1]-1)/2), int(round((ancpatches.shape[2]-1)/2))
        ancpatches = ancpatches[:,middrow,middcol]

        predhr = obj.predict(ancpatches, batch_size=batchsize)
        predhr = predhr.reshape(dissshape)
        return predhr

    elif mod == 'lenet':
        print('| --- Predicting new values, Le-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize)
        predhr = predhr.reshape(dissshape)
        return predhr

    elif mod == 'vgg':
        print('| --- Predicting new values, VGG-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize)
        predhr = predhr.reshape(dissshape)
        return predhr

    elif mod.endswith('unet'):
        if(mod == 'unet'):
            print('| --- Predicting new patches, 1 resolution U-Net')
            predhr = obj.predict(ancpatches, batch_size=batchsize)
            print('| ---- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, stride)
            return predhr

        elif mod.startswith('2r'):
            print('| --- Predicting new patches, 2 resolution U-Net')
            predhr = obj.predict([ancpatches, disspatcheso], batch_size=batchsize)[0]
            print('| ---- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, stride)
            return predhr

    else:
        print('Predict CNN - Unknown model')
