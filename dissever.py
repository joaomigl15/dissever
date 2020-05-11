import osgeoutils as osgu, nputils as npu, gputils as gput
import kerasutils as ku
import pycno, caret, neigPairs
import numpy as np
from sklearn import metrics
import metricsev as mev
import os


def runDissever(fshape, ancdatasets, yraster=None, rastergeo=None, perc2evaluate = 0.1, poly2agg = 'NUTSIII',
                method='lm', cnnmod='unet', patchsize = 7, epochspi=1, batchsize=1024, lrate=0.001, filters=[2,4,8,16,32],
                lweights=[1/2, 1/2], extdataset=None, p=1, min_iter=3, max_iter=100, converge=2,
                tempfileid=None, verbose=False):

    print('| DISSEVER')
    ymethod = yraster.split('/')[-1].split('_')[0]
    casestudy = 'belgium_' + ymethod + '_' + str(patchsize) + cnnmod + '_bs' + str(batchsize) + \
                '_lr' + str(lrate) + '_' + str(epochspi) + 'epochspi_10+10'

    if patchsize == 16:
        cstudyad = 'ghspgbuglctehsdfwtd16'
    elif patchsize == 32:
        cstudyad = 'ghspgbuglctehsdfwtd32'
    else:
        cstudyad = None

    nrowsds = ancdatasets[:,:,0].shape[1]
    ncolsds = ancdatasets[:,:,0].shape[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0] # ID's polígonos originais

    print('| Computing polygons areas')
    polygonareas = gput.computeAreas(fshape)

    if yraster:
        disseverdataset, rastergeo = osgu.readRaster(yraster)
        idpolvalues = npu.statsByID(disseverdataset, idsdataset, 'sum')
    else:
        polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
        idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
        disseverdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid)

    iterdissolv = False
    if method.startswith('ap'):
        if iterdissolv:
            adjpolygons = gput.computeNeighbors(fshape, polyg = poly2agg, verbose=True)
        else:
            adjpairs, newpairs = neigPairs.createNeigSF(fshape, polyg=poly2agg)

    dissmask = np.copy(idsdataset)
    dissmask[~np.isnan(dissmask)] = 1
    ancvarsmask = np.dstack([dissmask] * ancdatasets.shape[2])

    olddisseverdataset = disseverdataset

    if method.endswith('cnn'):
        # Create anc variables patches (includes replacing nans by 0, and 0 by nans)
        print('| Creating ancillary variables patches')
        ancdatasets[np.isnan(ancdatasets)] = 0
        padd = True if cnnmod == 'lenet' or cnnmod == 'uenc' or cnnmod == 'vgg' else False
        ancpatches = ku.createpatches(ancdatasets, patchsize, padding=padd, stride=1, cstudy=cstudyad)
        ancdatasets = ancdatasets * ancvarsmask

        # Compile model and save initial weights
        cnnobj = ku.compilecnnmodel(cnnmod, [patchsize, patchsize, ancdatasets.shape[2]], lrate,
                                    filters=filters, lweights=lweights)
        cnnobj.save_weights('models_' + casestudy + '.h5')


    strat = True
    if method.startswith('ap'):
        if iterdissolv:
            if strat:
                numpols = round((perc2evaluate/2) * (len(adjpolygons)))
                pquartilles = gput.polygonsQuartilles(idpolvalues, numpols)
                adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, strat=pquartilles, verbose=True)
            else:
                adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, verbose=True)

            initadjpairs = adjpairs
            if(verbose): print('Fixed adjacent pairs (' + str(len(initadjpairs)) + ') -', initadjpairs)

    lasterror = -np.inf
    lowesterror = np.inf
    history = []
    for k in range(1, max_iter+1):
        print('| - Iteration', k)

        if (k%10) == 0:
            epochspi = epochspi + 10

        previdpolvalues = idpolvalues # Original polygon values
        if method.startswith('ap'):
            if iterdissolv:
                if strat:
                    pquartilles = gput.polygonsQuartilles(idpolvalues, numpols)
                    adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate / 2, strat=pquartilles, initadjpairs=initadjpairs, verbose=True)
                else:
                    adjpairs = gput.createAdjPairs(adjpolygons, perc2evaluate/2, initadjpairs=initadjpairs)

                if (verbose): print('Adjacent pairs (' + str(len(adjpairs)) + ') -', adjpairs)
                newshape, newpairs = gput.dissolvePairs(fshape, adjpairs)

                osgu.removeShapefile(newshape)


            idsdataset2e = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]

            # Edit idpolvalues
            pairslist = [item for t in adjpairs for item in t]

            ids2keep = list(set(previdpolvalues.keys()))
            idpolvalues = dict((k, previdpolvalues[k]) for k in ids2keep)

            idpolvalues2e = dict((k, previdpolvalues[k]) for k in pairslist)
            polygonarea2e = dict((k, polygonareas[k]) for k in pairslist)


        if method.endswith('cnn'):
            print('| -- Updating dissever patches')
            disseverdataset[np.isnan(disseverdataset)] = 0
            disspatches = ku.createpatches(disseverdataset, patchsize, padding=padd, stride=1)
            disseverdataset = disseverdataset * dissmask

            print('| -- Fitting the model')
            historycnn = caret.fitcnn(ancpatches, disspatches, p, cnnmod=cnnmod,
                                      cnnobj=cnnobj, epochs=epochspi, batchsize=batchsize, extdataset=extdataset)
            loss = historycnn.history['loss']

            print('| -- Predicting new values')
            disseverdataset = caret.predictcnn(cnnobj, cnnmod, ancpatches, disspatches, disseverdataset.shape, batchsize=batchsize)

            # Reset model weights
            print('------- LOAD INITIAL WEIGHTS')
            cnnobj.load_weights('models_' + casestudy + '.h5')

        else:
            print('| -- Fitting the model')
            mod = caret.fit(ancdatasets, disseverdataset, p, method, batchsize, lrate, epochspi)

            print('| -- Predicting new values')
            # Replace NaN's by 0 and predict
            ancdatasets[np.isnan(ancdatasets)] = 0
            disseverdataset = caret.predict(mod, ancdatasets)
            disseverdataset = np.expand_dims(disseverdataset, axis=2)


        # Replace NaN zones by Nan
        disseverdataset = disseverdataset * dissmask
        disseverdataset[disseverdataset < 0] = 0
        disseverdataset2e = np.copy(disseverdataset)
        ancdatasets = ancdatasets * ancvarsmask
        davg, dsdev = np.nanmean(disseverdataset), np.nanstd(disseverdataset)

        if verbose: print('| -- Computing adjustement factor')
        stats = npu.statsByID(disseverdataset, idsdataset, 'sum')

        if method.startswith('ap'):
            stats2e = npu.statsByID(disseverdataset2e, idsdataset2e, 'sum')
            stats2e = dict((k, stats2e[k]) for k in pairslist)

        # Horrible hack, avoid division by 0
        for s in stats: stats[s] = stats[s] + 0.00001
        for s in stats2e: stats2e[s] = stats2e[s] + 0.00001

        polygonratios = {k: idpolvalues[k] / stats[k] for k in stats.keys() & idpolvalues}
        polygonratios2e = {k: idpolvalues2e[k] / stats2e[k] for k in stats2e.keys() & idpolvalues2e}
        idpolvalues = previdpolvalues

        for polid in polygonratios:
            disseverdataset[idsdataset == polid] = (disseverdataset[idsdataset == polid] * polygonratios[polid])

        for polid in polygonratios2e:
            disseverdataset2e[idsdataset2e == polid] = (disseverdataset2e[idsdataset2e == polid] * polygonratios2e[polid])


        osgu.writeRaster(disseverdataset[:,:,0], rastergeo, 'tempfiledissever_' + casestudy + '_' + str(k).zfill(2) + 'it.tif')

        if method.startswith('ap'):
            # Compute metrics for the evaluation municipalities
            actual2e = list(idpolvalues2e.values())
            predicted2e = list(stats2e.values())
            areas2e = list(polygonarea2e.values())
            range2e = max(actual2e) - min(actual2e)

            mae2e, wae2e = mev.mean_absolute_error(actual2e, predicted2e, areas2e)
            rmse2e = np.sqrt(metrics.mean_squared_error(actual2e, predicted2e))
            # nmae2e = metrics.mean_absolute_error(actual2e, predicted2e) / range2e
            # nrmse2e = np.sqrt(metrics.mean_squared_error(actual2e, predicted2e)) / range2e

            filenamemetrics2e = 'metrics2e_' + casestudy + '_2e.csv'
            if os.path.exists(filenamemetrics2e):
                with open(filenamemetrics2e, 'a') as myfile:
                    myfile.write(str(k) + ';' + str(mae2e) + ';' + str(rmse2e) + ';' + str(dsdev) + '\n')
            else:
                with open(filenamemetrics2e, 'w+') as myfile:
                    myfile.write('IT;MAE;RMSE;SDEV\n')
                    myfile.write(str(k) + ';' + str(mae2e) + ';' + str(rmse2e) + ';' + str(dsdev) + '\n')

        # Check if the algorithm converged
        error = np.nanmean(abs(disseverdataset-olddisseverdataset))
        history.append([error, loss]) if method.endswith('cnn') else history.append(error)
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
