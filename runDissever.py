import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import osgeoutils as osgu, dissever
import numpy as np
import tensorflow as tf
import os

from numpy.random import seed
from tensorflow import set_random_seed


indicator = 'LiveBirths18'
admboundary1 = 'nutsiii'
admboundary2 = 'municip'
admboundary2e = 'cparish'


method = ['lm']
cnnmodel = ['2runet']
psamples = [1]
epochspi = [1]
batchsize = [1024]
learningrate = [0.001]
extendeddataset = [None]
patchsize = [16]
filters = [[2,4,8,16,32]]
lossweights = [[0.1, 0.9]]


fshape = os.path.join('Shapefiles', (admboundary2 + '13_cont.shp'))
fshape2e = os.path.join('Shapefiles', (admboundary2e + '13wid_cont.shp'))
fcsv = os.path.join('Statistics', indicator, (admboundary2 + '.csv'))
yraster = os.path.join('Results', indicator, 'Baselines', ('td_200m' + '.tif'))


ancdataset1, rastergeo = osgu.readRaster('Rasters/Normalized/nl16_200m.tif')
ancdataset2 = osgu.readRaster('Rasters/Normalized/lc18_200m.tif')[0]
ancdataset3 = osgu.readRaster('Rasters/Normalized/dfw18_200m.tif')[0]
ancdataset4 = osgu.readRaster('Rasters/Normalized/dfr15_200m.tif')[0]
#ancdataset4 = osgu.readRaster('Rasters/Normalized/dfr15.tif')[0]
#ancdataset5 = osgu.readRaster('Rasters/Normalized/dfw18.tif')[0]

ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4))
ancnames = ['NL16', 'LC18', 'DFW18', 'DFR15']


if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, [admboundary1.upper(), admboundary2.upper()])

for psamp in psamples:
    for meth in method:
        if (meth != 'cnn'):
            print('\n--- Running dissever leveraging a', meth, 'model')
            seed(42)
            set_random_seed(42)

            dissdataset, rastergeo, history = dissever.runDissever(fshape, fshape2e, ancdatasets, ancnames=ancnames,
                                                                   max_iter=10,
                                                                   rastergeo=rastergeo, method=meth,
                                                                   p=psamp,
                                                                   yraster=yraster,
                                                                   verbose=True)

            print('- Writing raster to disk...')
            niter = len(history)-1
            osgu.writeRaster(dissdataset, rastergeo,
                             'dissever_tdnl_' + indicator + '_' + meth + '_psamp=' + str(psamp) + '_' + str(niter) + 'iter.tif')

            print('- Plotting error per iteration...')
            errorvalues = history[0:niter]
            plt.plot(list(range(1,niter+1)), errorvalues)
            plt.title(indicator)
            plt.ylabel('Error')
            plt.savefig('errorpi_td16_' + indicator + '_' + meth + '_psamp=' + str(psamp) + '_' + str(niter) + 'iter.png')
            plt.clf()

        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)

            for cnnm in cnnmodel:
                for lweights in lossweights:
                    for filt in filters:
                        for psize in patchsize:
                            for epiter in epochspi:
                                for bsize in batchsize:
                                    for lrate in learningrate:
                                        for edatas in extendeddataset:
                                            print('\n--- Running dissever with the following CNN configuration:')
                                            print('- Method:', cnnm,
                                                  '| Percentage of sampling:', psamp,
                                                  '| Epochs per iteration:', epiter,
                                                  '| Batch size:', bsize,
                                                  '| Learning rate:', lrate,
                                                  '| Filters:', filt,
                                                  '| Loss weights:', lweights,
                                                  '| Patch size:', psize,
                                                  '| Extended dataset:', edatas)

                                            seed(42)
                                            set_random_seed(42)

                                            dissdataset, rastergeo, history = dissever.runDissever(fshape, fshape2e, ancdatasets, ancnames=ancnames,
                                                                                                   min_iter=99, max_iter=100,
                                                                                                   rastergeo=rastergeo,
                                                                                                   method='cnn', cnnmod=cnnm,
                                                                                                   patchsize=psize, batchsize=bsize,
                                                                                                   epochspi=epiter, lrate=lrate,
                                                                                                   filters=filt, lweights=lweights,
                                                                                                   extdataset=edatas, p=psamp,
                                                                                                   yraster=yraster, yrasterlr=yrasterlr,
                                                                                                   converge=1.5, verbose=True)

                                            print('- Writing raster to disk...')
                                            niter = len(history)-1
                                            osgu.writeRaster(dissdataset, rastergeo,
                                                             'dissever_td16_' + indicator + '_' + cnnm
                                                             + '_bsize=' + str(bsize)
                                                             + '_epiter=' + str(epiter)
                                                             + '_lrate=' + str(lrate)
                                                             + '_psamp=' + str(psamp)
                                                             + '_filt=' + str(filt)
                                                             + '_lweights=' + str(lweights)
                                                             + '_psize=' + str(psize)
                                                             + '_edat=' + str(edatas)
                                                             + '_' + str(niter) + 'iter.tif')

                                            print('- Plotting loss per iteration...')
                                            lossvalues = [x[1] for x in history]
                                            lossvalues = lossvalues[0:niter]
                                            plt.plot(list(range(1, niter+1)), lossvalues)
                                            plt.title(indicator)
                                            plt.ylabel('Loss')
                                            plt.savefig('losspit_td16_' + indicator + '_' + cnnm
                                                        + '_bsize=' + str(bsize)
                                                        + '_epiter=' + str(epiter)
                                                        + '_lrate=' + str(lrate)
                                                        + '_psamp=' + str(psamp)
                                                        + '_filt=' + str(filt)
                                                        + '_lweights=' + str(lweights)
                                                        + '_psize=' + str(psize)
                                                        + '_edat=' + str(edatas)
                                                        + '_' + str(niter) + 'iter.png')
                                            plt.clf()

                                            print('- Plotting error per iteration...')
                                            errorvalues = [x[0] for x in history]
                                            errorvalues = errorvalues[0:niter]
                                            plt.plot(list(range(1, niter+1)), errorvalues)
                                            plt.title(indicator)
                                            plt.ylabel('Error')
                                            plt.savefig('errorpi_td16_' + indicator + '_' + cnnm
                                                        + '_bsize=' + str(bsize)
                                                        + '_epiter=' + str(epiter)
                                                        + '_lrate=' + str(lrate)
                                                        + '_psamp=' + str(psamp)
                                                        + '_filt=' + str(filt)
                                                        + '_lweights=' + str(lweights)
                                                        + '_psize=' + str(psize)
                                                        + '_edat=' + str(edatas)
                                                        + '_' + 'cnn' + '_' + str(niter) + 'iter.png')
                                            plt.clf()

if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
