import numpy as np
import osgeoutils as osgu


def runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=None):
    print('|| MASS-PRESERVING AREAL WEIGHTING')
    unique, counts = np.unique(idsdataset[~np.isnan(idsdataset)], return_counts=True)
    counts = dict(zip(unique, counts))

    for polid in counts:
        idsdataset[idsdataset == polid] = counts[polid]

    masspdataset = polygonvaluesdataset/idsdataset

    if tempfileid:
        tempfile = 'tempfilemp_' + tempfileid + '.tif'
        osgu.writeRaster(masspdataset[:, :, 0], rastergeo, tempfile)

    return masspdataset, rastergeo
