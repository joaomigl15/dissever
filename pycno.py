import numpy as np
from scipy import ndimage
import osgeoutils as osgu, nputils as npu
import massp



def runPycno(fshape, res, niter=30, converge=0.001):
    print('| PYCNOPHYLACTIC INTERPOLATION')
    idsdataset = osgu.readRaster(osgu.ogr2raster(fshape, res=res, attr='ID'))[0]
    polygonvaluesdataset, rastergeo = osgu.readRaster(osgu.ogr2raster(fshape, res=res, attr='VALUE'))

    fmpraster = massp.runMassPreserving(idsdataset, polygonvaluesdataset, rastergeo)
    pycnodataset = osgu.readRaster(fmpraster)[0]
    oldpycnodataset = pycnodataset

    idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)

    for it in range(1, niter+1):
        print('|| - Iteration', it)

        # Calculate the mean of the cells in the 3 by 3 neighborhood
        pycnodataset = ndimage.generic_filter(pycnodataset, np.nanmean, size=3, mode='constant', cval=np.NaN)

        # Summarizes the values within each polygon
        stats = npu.statsByID(pycnodataset, idsdataset, 'sum')

        # Divide the true polygon values by the estimated polygon values (= ratio)
        polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}

        # Multiply ratio by the different cells within each polygon
        for polid in polygonratios:
            if polid != -9999:
                pycnodataset[idsdataset == polid] = (pycnodataset[idsdataset == polid] * polygonratios[polid])

        # Check if the algorithm converged
        if((it > 1) and (np.mean(abs(pycnodataset-oldpycnodataset)) < converge)):
            break
        else:
            oldpycnodataset = pycnodataset


    tempfile = 'tempfilepycno.tif'
    osgu.writeRaster(pycnodataset, rastergeo, tempfile)
    return tempfile
