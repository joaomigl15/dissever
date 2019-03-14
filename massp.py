import numpy as np
from gith.dissever import osgeoutils as osgu


def runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=None):
    print('|| MASS-PRESERVING AREAL WEIGHTING')
    unique, counts = np.unique(idsdataset[~np.isnan(idsdataset)], return_counts=True)
    counts = dict(zip(unique, counts))

    for polid in counts:
        idsdataset[idsdataset == polid] = counts[polid]

    masspdataset = polygonvaluesdataset/idsdataset

    tempfile = 'tempfilemp_' + tempfileid + '.tif'
    if tempfileid: osgu.writeRaster(masspdataset, rastergeo, tempfile)
    return masspdataset, rastergeo
