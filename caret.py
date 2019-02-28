from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def fitlm(X, y):
    mod = LinearRegression()
    mod = mod.fit(X, y)
    return mod


def fitrf(X, y):
    mod = RandomForestRegressor()
    mod = mod.fit(X, y)
    return mod


def fit(X, y, method):
    X = X.reshape((-1, X.shape[2]))
    y = np.ravel(y)
    if(method == 'lm'):
        return fitlm(X, y)
    elif(method == 'rf'):
        return fitrf(X, y)
    else:
        return None


def predict(mod, X):
    if(mod != 'cnn'):
        newX = X.reshape((-1, X.shape[2]))
        pred = mod.predict(newX)
        pred = pred.reshape(X.shape[0], X.shape[1])
        return pred
    else:
        return mod.predict(X)


def fitcnn(X, y, cnnmod, epochs, batchsize):
    return cnnmod.fit(X, y, epochs=epochs, batch_size=batchsize)
