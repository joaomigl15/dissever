import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import osgeoutils as osgu
import dissever
import numpy as np
import os

from numpy.random import seed


indicator = 'Belgium'
admboundary2 = 'ADMDISTRIC'

method = ['apcnn'] # aplm, apcnn,
cnnmodel = 'unet' # lenet, vgg8, vgg, unet, 2runet
psamples = [0.25] # 0.25, 1
epochspi = [1]
batchsizearr = [64] # 256, 1024
learningrate = [0.0001] # 0.01, 0.001, census-0.0001
extendeddataset = [None] # None, '2.5T6'
lossweights = [[0.1, 0.9]]
percs2evaluate = [1]
ymethod = 'pycno' # massp, pycno, td, lm5i

if cnnmodel == 'lenet':
    print('Lenet')
    filters = [[14,28,56,112,224]]
    patchsize = [7]
elif cnnmodel == 'vgg':
    print('VGG')
    filters = [[8,16,32,64,512]] # vgg[64,128,256,512,4096]
    patchsize = [32]
elif cnnmodel == 'unet' or cnnmodel == 'uenc':
    print('Unet')
    filters = [[8,16,32,64,128]]
    patchsize = [16]
else:
    filters = [[14,28,56,112,224]]
    patchsize = [7]

fshape = os.path.join('Shapefiles', indicator, (admboundary2 + '.shp'))
fcsv = os.path.join('Statistics', indicator, (admboundary2 + '.csv'))
yraster = os.path.join('Results', indicator, 'Baselines', (ymethod + '_200m.tif'))

ancdataset1, rastergeo = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'ghspg_1975_200m.tif'))
ancdataset2 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'bug_1975_200m.tif'))[0]
ancdataset3 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'lc_1900_200m.tif'))[0]
ancdataset4 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'te_2018_200m.tif'))[0]
ancdataset5 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'hs_2016_200m.tif'))[0]
ancdataset6 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'dfw_1900_200m.tif'))[0]
ancdataset7 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'td_200m.tif'))[0]

# ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5, ancdataset6))
ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5, ancdataset6, ancdataset7))


if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
# if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, [admboundary1.upper(), admboundary2.upper()])
if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, admboundary2.upper())

for psamp in psamples:
    for meth in method:
        for p2e in percs2evaluate:
            for batchsize in batchsizearr:
                for epiter in epochspi:
                    for lrate in learningrate:
                        if not meth.endswith('cnn'):
                            print('\n--- Running dissever leveraging a', meth, 'model')
                            seed(42)

                            casestudy = indicator + '_' + ymethod + '_' + str(psize) + method + '_bs' + \
                                        str(batchsize) + '_lr' + str(lrate) + '_' + str(epiter) + \
                                        'epochspi'
                            dissdataset, rastergeo, history = dissever.runDissever(fshape, ancdatasets,
                                                                                   min_iter=49, max_iter=50,
                                                                                   perc2evaluate=p2e, poly2agg=admboundary2.upper(),
                                                                                   rastergeo=rastergeo, method=meth,
                                                                                   batchsize=batchsize,
                                                                                   epochspi=epiter, lrate=lrate,
                                                                                   p=psamp,
                                                                                   yraster=yraster,
                                                                                   verbose=True)

                            print('- Writing raster to disk...')
                            niter = len(history)-1
                            osgu.writeRaster(dissdataset, rastergeo,
                                             'dissever_tdallp_' + str(p2e) + '_'
                                             + indicator + '_' + meth + '_psamp=' + str(psamp) + '_' + str(niter) + 'iter.tif')

                            print('- Plotting error per iteration...')
                            errorvalues = history[0:niter]
                            plt.plot(list(range(1,niter+1)), errorvalues)
                            plt.title(indicator)
                            plt.ylabel('Error')
                            plt.savefig('errorpi_tdallp_' + str(p2e)
                                        + '_' + indicator
                                        + '_' + meth
                                        + '_bsize=' + str(batchsize)
                                        + '_psamp=' + str(psamp) +
                                        '_' + str(niter) + 'iter.png')
                            plt.clf()

                        else:
                            for lweights in lossweights:
                                for filt in filters:
                                    for psize in patchsize:
                                        for edatas in extendeddataset:
                                            print('\n--- Running dissever with the following CNN configuration:')
                                            print('- Method:', cnnmodel,
                                                  '| Percentage of sampling:', psamp,
                                                  '| Epochs per iteration:', epiter,
                                                  '| Batch size:', batchsize,
                                                  '| Learning rate:', lrate,
                                                  '| Filters:', filt,
                                                  '| Loss weights:', lweights,
                                                  '| Patch size:', psize,
                                                  '| Extended dataset:', edatas)

                                            import tensorflow as tf
                                            seed(42)
                                            tf.random.set_seed(42)
                                            # config = tf.compat.v1.ConfigProto()
                                            # config.gpu_options.allow_growth = True
                                            # session = tf.compat.v1.Session(config=config)

                                            casestudy = indicator + '_' + ymethod + '_' + str(psize) + cnnmodel + \
                                                '_bs' + str(batchsize) + '_lr' + str(lrate) + '_' + str(epiter) + \
                                                'epochspi_10+10'
                                            dissdataset, rastergeo, history = dissever.runDissever(fshape, ancdatasets,
                                                                                                   min_iter=49, max_iter=50,
                                                                                                   perc2evaluate=p2e,
                                                                                                   poly2agg=admboundary2.upper(),
                                                                                                   rastergeo=rastergeo,
                                                                                                   method=meth, cnnmod=cnnmodel,
                                                                                                   patchsize=psize, batchsize=batchsize,
                                                                                                   epochspi=epiter, lrate=lrate,
                                                                                                   filters=filt, lweights=lweights,
                                                                                                   extdataset=edatas, p=psamp,
                                                                                                   yraster=yraster,
                                                                                                   converge=1.5, verbose=True)

                                            print('- Writing raster to disk...')
                                            niter = len(history)-1
                                            osgu.writeRaster(dissdataset, rastergeo,
                                                             'dissever_' + casestudy + '_'
                                                             + indicator + '_' + cnnmodel
                                                             + '_bsize=' + str(batchsize)
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
                                            plt.savefig('losspit_tdallp_' + casestudy + '_'
                                                        + indicator + '_' + cnnmodel
                                                        + '_bsize=' + str(batchsize)
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
                                            plt.savefig('errorpi_tdallp_' + casestudy + '_'
                                                        + indicator + '_' + cnnmodel
                                                        + '_bsize=' + str(batchsize)
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
