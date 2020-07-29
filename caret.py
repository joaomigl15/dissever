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
from numpy import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

SEED = 42


def fitlm(X, y):
    mod = LinearRegression()
    mod = mod.fit(X, y)
    return mod

def fitsgdregressor(X, y, batchsize, lrate, epoch):
    mod = SGDRegressor(max_iter=epoch, alpha=0, learning_rate='constant', eta0=lrate, verbose=1)
    mod = mod.fit(X, y)
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

    relevantids = np.where(~np.isnan(y))[0]
    relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
    idsamples = relevantids[relevsamples]
    print('| --- Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

    X = X[idsamples,:]
    y = y[idsamples]
    # y = np.log(1+y)

    if(method.endswith('lm')):
        print('|| Fit: Linear Model')
        return fitlm(X, y)
    if (method.endswith('sgdregressor')):
        print('|| Fit: SGD Regressor')
        return fitsgdregressor(X, y, batchsize, lrate, epoch)
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
        # ReduceLROnPlateau(monitor='loss', min_delta=0.01, patience=3, factor=0.1, min_lr=5e-6, verbose=1),
        # EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=1, restore_best_weights=True)
        EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=1, restore_best_weights=True)
    ]

def generator(X, y, batch_size):
     # Create empty arrays to contain batch of features and labels
     batch_features = np.zeros((batch_size, X.shape[1], X.shape[2], X.shape[3]))
     batch_labels = np.zeros((batch_size, y.shape[1], y.shape[2], y.shape[3]))
     while True:
         for i in range(batch_size):
             # choose random index in features
             index = random.choice(y.shape[0], 1)
             batch_features[i] = X[index,:,:,:]
             batch_labels[i] = y[index,:,:,:]
         yield batch_features, batch_labels


def fitcnn(X, y, p, cnnmod, cnnobj, casestudy, epochs, batchsize, extdataset):
    tf.random.set_seed(SEED)

    # Reset model weights
    print('------- LOAD INITIAL WEIGHTS')
    cnnobj.load_weights('Temp/models_' + casestudy + '.h5')

    if cnnmod == 'lenet':
        print('| --- Fit - 1 resolution Le-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            Xfit = X[idsamples, :, :, :]
            yfit = y[idsamples]
            hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize)]
            return hislist

    elif cnnmod == 'vgg':
        print('| --- Fit - 1 resolution VGG-Net')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
        relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if extdataset:
            newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            return cnnobj.fit(X[idsamples, :, :, :], y[idsamples], epochs=epochs, batch_size=batchsize)

    elif cnnmod == 'uenc':
        print('| --- Fit - 1 resolution U-Net encoder')

        # Select midd pixel from patches
        middrow, middcol = int((y.shape[1]-1)/2), int(round((y.shape[2]-1)/2))
        y = y[:,middrow,middcol,0]

        relevantids = np.where(~np.isnan(y))[0]
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

        # Train only with patches having middpixel different from NaN
        relevantids = np.where(~np.isnan(y[:,middrow,middcol,0]))[0]

        if len(p) > 1:
            relevsamples = np.random.choice(len(relevantids), round(len(relevantids)), replace=False)
        else:
            relevsamples = np.random.choice(len(relevantids), round(len(relevantids) * p[0]), replace=False)
        idsamples = relevantids[relevsamples]
        print('Number of instances (All, >0, X%>0):', y.shape[0], len(relevantids), len(idsamples))

        if(cnnmod == 'unet'):
            if extdataset:
                newX, newy = npu.extenddataset(X[idsamples, :, :, :], y[idsamples], transf=extdataset)
                return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
            else:
                if len(p) > 1:
                    print('| --- Fit-generator - 1 resolution U-Net Model')
                    hislist = []
                    Xgen = X[idsamples, :, :, :]
                    ygen = y[idsamples]
                    for i in range(len(p)):
                        cnnobj.load_weights('models_' + casestudy + '.h5')
                        mod = cnnobj.fit(generator(Xgen, ygen, batchsize),
                                         steps_per_epoch=round(len(idsamples) * p[i] / batchsize), epochs=epochs)
                        hislist.append(mod)
                        cnnobj.save_weights('snapshot_' + casestudy + '_' + str(i) + '.h5')
                    return hislist
                else:
                    print('| --- Fit - 1 resolution U-Net Model')
                    Xfit = X[idsamples, :, :, :]
                    yfit = y[idsamples]
                    # yfit = np.log(1+yfit)
                    yfit[np.isnan(yfit)] = 0
                    hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize)]
                    # hislist = [cnnobj.fit(Xfit, yfit, epochs=epochs, batch_size=batchsize, callbacks=get_callbacks())]
                    return hislist

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
    # pred = np.exp(pred)-1
    pred = pred.reshape(X.shape[0], X.shape[1])
    return [pred]

def predictcnn(obj, mod, fithistory, casestudy, ancpatches, dissshape, batchsize, stride=1):
    if mod == 'lenet':
        print('| --- Predicting new values, Le-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod == 'vgg':
        print('| --- Predicting new values, VGG-Net')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod == 'uenc':
        print('| --- Predicting new values, U-Net encoder')
        predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
        predhr = predhr.reshape(dissshape)
        return [predhr]

    elif mod.endswith('unet'):
        if(mod == 'unet'):
            if len(fithistory) > 1:
                print('| --- Predicting new patches from several models, 1 resolution U-Net')
                predhr = []
                for i in range(len(fithistory)):
                    obj.load_weights('snapshot_' + casestudy + '_' + str(i) + '.h5')
                    predhr.append(obj.predict(ancpatches, batch_size=batchsize, verbose=1))
                print('| ---- Reconstructing HR images from patches..')
                for i in range(len(predhr)): predhr[i] = ku.reconstructpatches(predhr[i], dissshape, stride)
                return predhr
            else:
                print('| --- Predicting new patches, 1 resolution U-Net')
                predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)
                print('| ---- Reconstructing HR image from patches..')
                predhr = ku.reconstructpatches(predhr, dissshape, stride)
                # predhr = np.exp(predhr)-1
                return [predhr]

        elif mod.startswith('2r'):
            print('| --- Predicting new patches, 2 resolution U-Net')
            predhr = obj.predict(ancpatches, batch_size=batchsize, verbose=1)[0]
            print('| ---- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, stride)
            return predhr

    else:
        print('Predict CNN - Unknown model')
