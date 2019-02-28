import osgeoutils as osgu, numpy as np


def runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo):
    print('|| MASS-PRESERVING AREAL WEIGHTING')
    unique, counts = np.unique(idsdataset, return_counts=True)
    counts = dict(zip(unique, counts))

    for polid in counts:
        if polid != -9999:
            idsdataset[idsdataset == polid] = counts[polid]

    masspdataset = polygonvaluesdataset/idsdataset

    tempfile = 'tempfilemp.tif'
    osgu.writeRaster(masspdataset, rastergeo, tempfile)
    return tempfile
