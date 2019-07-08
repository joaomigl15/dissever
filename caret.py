#from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import nputils as npu, kerasutils as ku


def fitlm(X, y):
    mod = LinearRegression()
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



def fit(X, y, idsamples, method):
    X = X.reshape((-1, X.shape[2]))[idsamples]
    y = np.ravel(y)[idsamples]
    if(method.endswith('lm')):
        return fitlm(~np.isnan(X), ~np.isnan(y))
    elif(method.endswith('rf')):
        return fitrf(~np.isnan(X), ~np.isnan(y))
    elif(method.endswith('xgbtree')):
        return fitxgbtree(~np.isnan(X), ~np.isnan(y))
    else:
        return None


def fitcnn(X, y, yo, idsamples, cnnmod, cnnobj, epochs, batchsize, extdataset):
    ids = np.where(np.any(~np.isnan(X[idsamples, :, :, 0]), axis=(1, 2)))[0]
    if cnnmod != 'lenet':
        if cnnmod.startswith('2r'): # e.g., 2runet
            print('| --- 2 resolutions Model')
            if extdataset:
                newX, newy, newylr = npu.extenddataset(X[ids, :, :, :], y[ids], ylr[ids], transf=extdataset)
                newy = [newy, newy] #MUDOU
            else:
                newX = [X[ids, :, :, :], yo[ids]]
                newy = [y[ids], np.zeros((y.shape[0], 1, 1, 1))] #MUDOU
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        elif cnnmod.startswith('3r'): # e.g., 2runet
            print('| --- 2 resolutions Model')
            if extdataset:
                newX, newy, newylr = npu.extenddataset(X[ids, :, :, :], y[ids], ylr[ids], transf=extdataset)
                newy = [newy, newy] #MUDOU
            else:
                newX = [X[ids, :, :, :], yo[ids]]
                newy = [y[ids], np.zeros((y.shape[0], 4, 4, 1)), np.zeros((y.shape[0], 2, 2, 1))] #MUDOU
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
        else:
            print('| --- 1 resolution Model')
            if extdataset:
                newX, newy = npu.extenddataset(X[ids, :, :, :], y[ids], transf=extdataset)
            else:
                newX = X[ids, :, :, :]
                newy = y[ids]
            return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)
    else:
        print('| --- 1 resolution Le-Net')
        y = np.ravel(y)
        if extdataset:
            newX, newy = npu.extenddataset(X[ids, :, :, :], y[ids], transf=extdataset)
        else:
            newX = X[ids, :, :, :]
            newy = y[ids]
        return cnnobj.fit(newX, newy, epochs=epochs, batch_size=batchsize)


def predict(mod, X):
    newX = X.reshape((-1, X.shape[2]))
    pred = mod.predict(newX)
    pred = pred.reshape(X.shape[0], X.shape[1])
    return pred

def predictcnn(obj, mod, ancpatches, disspatcheso, dissshape, batchsize, stride=1):
    if mod != 'lenet':
        # midpixel = round(patchsize / 2)
        if mod.startswith('2r') or mod.startswith('3r'):
            print('| --- Predicting new patches..')
            predhr = obj.predict([ancpatches, disspatcheso], batch_size=batchsize)[0]
            print('| --- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, stride)
            #print('| --- Reconstructing LR image from patches..')
            #predlr = ku.reconstructpatches(predlr, (int(dissshape[0]/4), int(dissshape[1]/4), dissshape[2]), stride)
            return predhr
        else:
            print('| --- Predicting new patches..')
            predhr = obj.predict(ancpatches, batch_size=batchsize)
            print('| --- Reconstructing HR image from patches..')
            predhr = ku.reconstructpatches(predhr, dissshape, 1)
            return predhr
    else:
        print('| --- Predicting new values..')
        predhr = obj.predict(ancpatches, batch_size=batchsize)
        predhr = predhr.reshape(dissshape)
        return predhr
