from gith.dissever import osgeoutils as osgu, kerasutils as ku, nputils as npu, pycno, caret
import numpy as np


def runDissever(fshape, ancdatasets, ancnames, yraster=None, res=0.00166667,
                method='lm', cnnmodel='lenet', patchsize = 7, epochspi=1, batchsize=1024,
                p=None, min_iter=5, max_iter=100, converge=0.001,
                tempfileid=None, verbose=False):

    print('| DISSEVER')

    idsdataset = osgu.ogr2raster(fshape, res=res, attr='ID')[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, res=res, attr='VALUE')
    idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
    datasetmask = ~np.isnan(idsdataset)

    if yraster:
        disseverdataset, rastergeo = osgu.readRaster(yraster)
    else:
        disseverdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)

    olddisseverdataset = disseverdataset

    ancdatasets[np.isnan(ancdatasets)] = 0
    disseverdataset[np.isnan(disseverdataset)] = 0

    if(method == 'cnn'):
        cnnmodel = ku.compilecnnmodel(cnnmodel)
        ancpatches = ku.createpatches(ancdatasets, patchsize)
        disspatches = ku.createpatches(disseverdataset, patchsize)


    for k in range(1, max_iter+1):
        if(verbose): print('| - Iteration', k)

        if(method != 'cnn'):
            mod = caret.fit(ancdatasets, disseverdataset, method)
        else:
            mod = caret.fitcnn(ancpatches, disspatches, cnnmodel, epochs=epochspi, batchsize=batchsize)

        disseverdataset = caret.predict(mod, ancdatasets)

        if(verbose): print('| -- Computing adjustement factor')
        stats = npu.statsByID(disseverdataset, idsdataset, 'sum')
        polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}
        for polid in polygonratios:
            disseverdataset[idsdataset == polid] = (disseverdataset[idsdataset == polid] * polygonratios[polid])


        # Check if the algorithm has converged
        if((k >= min_iter) and (np.mean(abs(disseverdataset-olddisseverdataset)) < converge)):
            break
        else:
            olddisseverdataset = disseverdataset


    print(disseverdataset.shape)
    print(datasetmask.shape)
    print(disseverdataset)
    disseverdataset = disseverdataset * datasetmask
    print(disseverdataset)

    if tempfileid:
        tempfile = 'tempfiledissever_' + tempfileid + '.tif'
        osgu.writeRaster(disseverdataset, rastergeo, tempfile)

    return tempfile
