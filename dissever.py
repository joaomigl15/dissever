import osgeoutils as osgu, kerasutils as ku, nputils as npu, gputils as gput
import pycno, caret
import numpy as np
from sklearn import metrics


def runDissever(fshape, ancdatasets, ancnames, yraster=None, rastergeo=None, perc2evaluate = 0.10,
                method='lm', cnnmod='unet', patchsize = 7, epochspi=1, batchsize=1024, lrate=0.001, filters=[2,4,8,16,32],
                lweights=[1/2, 1/2], extdataset=None, p=1, min_iter=3, max_iter=100, converge=2,
                tempfileid=None, verbose=False):

    print('| DISSEVER')
    nrowsds = ancdatasets[:,:,0].shape[1]
    ncolsds = ancdatasets[:,:,0].shape[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]

    if method.startswith('ap'): adjmunicipalities = gput.computeNeighbors(fshape)

    if yraster:
        disseverdataset, rastergeo = osgu.readRaster(yraster)
        idpolvalues = npu.statsByID(disseverdataset, idsdataset, 'sum')
    else:
        polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
        idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
        disseverdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)

    dissmask = np.copy(idsdataset)
    dissmask[~np.isnan(dissmask)] = 1
    ancvarsmask = np.dstack([dissmask] * ancdatasets.shape[2])

    olddisseverdataset = disseverdataset

    if method == 'cnn':
        # Replace NaN's by 0
        ancdatasets[np.isnan(ancdatasets)] = 0
        disseverdataset[np.isnan(disseverdataset)] = 0

        padd = True if cnnmod == 'lenet' else False
        cnnobj = ku.compilecnnmodel(cnnmod, [patchsize, patchsize, ancdatasets.shape[2]], lrate,
                                    filters=filters, lweights=lweights)

        if cnnmod.startswith('2r'):
            ancpatches = ku.createpatches(ancdatasets, patchsize, padding=padd, stride=1)
            disspatches = ku.createpatches(disseverdataset, patchsize, padding=padd, stride=1, disseverds=True)
        else:
            ancpatches = ku.createpatches(ancdatasets, patchsize, padding=padd, stride=1)
            disspatches = ku.createpatches(disseverdataset, patchsize, padding=padd, stride=1, disseverds=True)
        disspatcheso = disspatches

        # Replace NaN zones by Nan
        disseverdataset = disseverdataset * dissmask
        ancdatasets = ancdatasets * ancvarsmask


    lasterror = -np.inf
    lowesterror = np.inf
    history = []
    for k in range(1, max_iter+1):
        if verbose: print('| - Iteration', k)

        previdpolvalues = idpolvalues
        if method.startswith('ap'):
            adjpairs = gput.createAdjPairs(adjmunicipalities, perc2evaluate/2)
            newshape, newpairs = gput.dissolvePairs(fshape, adjpairs)
            idsdataset = osgu.ogr2raster(newshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
            idsdataset2e = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
            osgu.removeShapefile(newshape)

            #Edit idpolvalues
            pairslist = [item for t in adjpairs for item in t]
            ids2keep = list(set(previdpolvalues.keys()) - set(pairslist))
            idpolvalues = dict((k, previdpolvalues[k]) for k in ids2keep)
            idpolvalues2e = dict((k, previdpolvalues[k]) for k in pairslist)
            for newpair in newpairs:
                idpolvalues[newpair] = previdpolvalues[newpairs[newpair][0]] + previdpolvalues[newpairs[newpair][1]]


        if method.endswith('cnn'):
            npatches = ancpatches.shape[0]
            nsamples = round(npatches * p)
            idsamples = np.random.choice(npatches, nsamples, replace=False)

            print('| -- Fitting the model')
            historycnn = caret.fitcnn(ancpatches, disspatches, disspatcheso, idsamples, cnnmod=cnnmod,
                                      cnnobj=cnnobj, epochs=epochspi, batchsize=batchsize, extdataset=extdataset)
            loss = historycnn.history['loss']

            print('| -- Predicting new values')
            disseverdataset = caret.predictcnn(cnnobj, cnnmod, ancpatches, disspatches, disseverdataset.shape, batchsize=batchsize)

        else:
            disssize = disseverdataset.shape[0] * disseverdataset.shape[1]
            nsamples = round(disssize * p)
            idsamples = np.random.choice(disssize, nsamples, replace=False)

            print('| -- Fitting the model')
            mod = caret.fit(ancdatasets, disseverdataset, idsamples, method)

            print('| -- Predicting new values')
            # Replace NaN's by 0 and predict
            ancdatasets[np.isnan(ancdatasets)] = 0
            disseverdataset = caret.predict(mod, ancdatasets)
            disseverdataset = np.expand_dims(disseverdataset, axis=2)


        # Replace NaN zones by Nan
        disseverdataset = disseverdataset * dissmask
        ancdatasets = ancdatasets * ancvarsmask

        # Avoid negative counts
        disseverdataset[disseverdataset < 0] = 0

        if verbose: print('| -- Computing adjustement factor')
        stats = npu.statsByID(disseverdataset, idsdataset, 'sum')
        if method.startswith('ap'):
            stats2e = npu.statsByID(disseverdataset, idsdataset2e, 'sum')
            stats2e = dict((k, stats2e[k]) for k in pairslist)


        # Horrible hack, avoid division by 0
        for s in stats: stats[s] = stats[s] + 0.00001

        polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}
        idpolvalues = previdpolvalues

        #osgu.writeRaster(disseverdataset[:,:,0], rastergeo, 'tempfiledisseverwadj_' + '_' + str(k) + 'it.tif')

        #print('** Mean of adj. factor', np.array(list(polygonratios.values())).mean())
        for polid in polygonratios:
            disseverdataset[idsdataset == polid] = (disseverdataset[idsdataset == polid] * polygonratios[polid])


        #osgu.writeRaster(disseverdataset[:,:,0], rastergeo, 'tempfiledissever_' + str(lrate) + '_' + str(k) + 'it.tif')

        if method.startswith('ap'):
            # Compute disaggregation error for 10% of the municipalities
            actual2e = list(idpolvalues2e.values())
            predicted2e = list(stats2e.values())
            range2e = max(actual2e) - min(actual2e)
            error2e = np.sqrt(metrics.mean_squared_error(actual2e, predicted2e)) / range2e
            print(error2e)


        # Check if the algorithm has converged
        error = np.nanmean(abs(disseverdataset-olddisseverdataset))
        history.append([error, loss]) if method == 'cnn' else history.append(error)
        errorrat = (error/lasterror) if lasterror>0 else np.inf
        lasterror = error
        print('Error:', error)

        if k >= min_iter:
            if errorrat < converge:
                if error < lowesterror:
                    lowesterror = error
                    lowesterriter = k
                    lowesterrdisseverdataset = np.copy(disseverdataset)
            else:
                if k == min_iter:
                    lowesterriter = k
                else:
                    disseverdataset = lowesterrdisseverdataset
                print('Retaining model fitted at iteration', lowesterriter)
                break
        olddisseverdataset = disseverdataset


    if tempfileid:
        tempfile = 'tempfiledissever_' + tempfileid + '.tif'
        osgu.writeRaster(disseverdataset, rastergeo, tempfile)

    return disseverdataset[:,:,0], rastergeo, history
