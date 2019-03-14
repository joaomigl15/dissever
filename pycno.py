import numpy as np
from scipy import ndimage
from gith.dissever import massp, osgeoutils as osgu, nputils as npu


def runPycno(idsdataset, polygonvaluesdataset, rastergeo, niter=100, converge=0.001, tempfileid=None):
    print('| PYCNOPHYLACTIC INTERPOLATION')

    pycnodataset = massp.runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)[0]
    oldpycnodataset = pycnodataset

    idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)

    for it in range(1, niter+1):
        print('| - Iteration', it)

        # Calculate the mean of the cells in the 3 by 3 neighborhood
        pycnodataset = ndimage.generic_filter(pycnodataset, np.nanmean, size=3, mode='constant', cval=np.NaN)

        # Summarizes the values within each polygon
        stats = npu.statsByID(pycnodataset, idsdataset, 'sum')

        # Divide the true polygon values by the estimated polygon values (= ratio)
        polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}

        # Multiply ratio by the different cells within each polygon
        for polid in polygonratios:
            pycnodataset[idsdataset == polid] = (pycnodataset[idsdataset == polid] * polygonratios[polid])

        # Check if the algorithm has converged
        if((it > 1) and (np.mean(abs(pycnodataset-oldpycnodataset)) < converge)):
            break
        else:
            oldpycnodataset = pycnodataset


    tempfile = 'tempfilepycno_' + tempfileid + '.tif'
    if tempfileid: osgu.writeRaster(pycnodataset, rastergeo, tempfile)
    return pycnodataset, rastergeo
