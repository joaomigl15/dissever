import numpy as np

def mean_absolute_error(actual, predicted, areas):
    abserror = np.abs(np.array(actual) - np.array(predicted))
    areaweig = (1/np.array(areas)) / (sum(1/np.array(areas)))
    # areaweig = (areaweig ** 3)
    wmae = abserror * areaweig
    return np.sum(wmae) / np.sum(areaweig), wmae
