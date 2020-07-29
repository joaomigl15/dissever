import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import osgeoutils as osgu
import dissever
import numpy as np, geopandas as gpd
import itertools, random

SEED = 42
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Turn off GPU

indicator = 'Withdrawals'
admboundary2 = 'NUTSIII'

methodopts = ['apcnn'] # aplm, apcnn
cnnmodelopts = ['unet'] # lenet, vgg, uenc, unet, 2runet
psamplesopts = [[1]] # [0.0625], [[0.0625], [0.03125]], [[0.03125, 0.0625]]]
epochspiopts = [10]
batchsizeopts = [64] # 256, 1024
learningrateopts = [0.001] # 0.01, 0.001, census-0.0001
extendeddatasetopts = [None] # None, '2.5T6'
lossweightsopts = [[0.1, 0.9]]
perc2evaluateopts = [1]
ymethodopts = ['td1pycno'] # pycno, td, tdnoise10, td1pycno, average25p75td

fshapea = os.path.join('Shapefiles', indicator, (admboundary2 + '.shp'))
fcsv = os.path.join('Statistics', indicator, (admboundary2 + '.csv'))

ancdataset1, rastergeo = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'ghspg_2015_200m.tif'))
ancdataset2 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'bua_2018_200m.tif'))[0]
ancdataset3 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'lc_2018_200m.tif'))[0]
ancdataset4 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'te_2020_200m.tif'))[0]
ancdataset5 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'hs_2012_200m.tif'))[0]
ancdataset6 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'dfw_2018_200m.tif'))[0]
ancdataset7 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'td_200m.tif'))[0]
ancdataset8 = osgu.readRaster(os.path.join('Rasters', indicator, 'Normalized', 'nl_2016_200m.tif'))[0]

ancdatasets1 = np.dstack((ancdataset1, ancdataset5, ancdataset2, ancdataset3))
ancdatasets2 = np.dstack((ancdataset1, ancdataset5, ancdataset2, ancdataset3, ancdataset8))
ancdatasetsopts = [ancdatasets2]


fshape = osgu.copyShape(fshapea, 'dissever')
# if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, [admboundary1.upper(), admboundary2.upper()])
if not ymethodopts: osgu.addAttr2Shapefile(fshape, fcsv, admboundary2.upper())

for (ancdts, psamples, method, perc2evaluate, ymethod) in itertools.product(ancdatasetsopts,
                                                                            psamplesopts,
                                                                            methodopts,
                                                                            perc2evaluateopts,
                                                                            ymethodopts):
    if not method.endswith('cnn'):
        print('\n--- Running dissever leveraging a', method, 'model')
        # seed(42)

        yraster = os.path.join('Results', indicator, 'Baselines', (ymethod + '_200m.tif'))
        casestudy = indicator + '_' + ymethod + '_' + method + '_p' + str(psamples) + '_' + str(ancdts.shape[2]) + 'va'
        dissdataset, rastergeo = dissever.runDissever(fshape, ancdts, min_iter=19, max_iter=20,
                                                      perc2evaluate=perc2evaluate, poly2agg=admboundary2.upper(),
                                                      rastergeo=rastergeo, method=method, p=psamples,
                                                      yraster=yraster, casestudy=casestudy,
                                                      verbose=True)

        print('- Writing raster to disk...')
        osgu.writeRaster(dissdataset, rastergeo, 'dissever_' + casestudy + '.tif')

    else:
        for cnnmodel in cnnmodelopts:
            if cnnmodel == 'lenet':
                print('Lenet')
                filtersopts = [[14, 28, 56, 112, 224]]
                patchsizeopts = [7]
            elif cnnmodel == 'vgg':
                print('VGG')
                filtersopts = [[8, 16, 32, 64, 512]]
                patchsizeopts = [32]
            elif cnnmodel.endswith('unet'):
                print('U-Net')
                filtersopts = [[8, 16, 32, 64, 128]]
                patchsizeopts = [16]
            elif cnnmodel == 'uenc':
                print('U-Net Encoder')
                filtersopts = [[8, 16, 32, 64, 128]]
                patchsizeopts = [16]
            else:
                filtersopts = [[14, 28, 56, 112, 224]]
                patchsizeopts = [7]

            for (lossweights, batchsize, epochpi, learningrate,
                 filters, patchsize, extendeddataset) in itertools.product(lossweightsopts,
                                                                           batchsizeopts,
                                                                           epochspiopts,
                                                                           learningrateopts,
                                                                           filtersopts,
                                                                           patchsizeopts,
                                                                           extendeddatasetopts):
                print('\n--- Running dissever with the following CNN configuration:')
                print('- Method:', cnnmodel, '| Percentage of sampling:', psamples,
                      '| Epochs per iteration:', epochpi, '| Batch size:', batchsize,
                      '| Learning rate:', learningrate, '| Filters:', filters,
                      '| Loss weights:', lossweights, '| Patch size:', patchsize,
                      '| Extended dataset:', extendeddataset)

                random.seed(SEED)
                np.random.seed(SEED)
                import tensorflow as tf
                # tf.compat.v1.disable_eager_execution()
                tf.keras.backend.clear_session()
                tf.random.set_seed(SEED)
                # tf.compat.v1.set_random_seed(SEED)
                # config = tf.compat.v1.ConfigProto()
                # config.gpu_options.allow_growth = True
                # session = tf.compat.v1.Session(config=config)

                yraster = os.path.join('Results', indicator, 'Baselines', (ymethod + '_200m.tif'))
                casestudy = indicator + '_' + ymethod + '_' + str(patchsize) + cnnmodel + '_bs' + str(batchsize) + \
                            '_p' + str(psamples) + '_' + str(epochpi) + 'epochspi' + '_' + str(ancdts.shape[2]) + 'va-unetconcatinplm-bll-sl1'
                dissdataset, rastergeo = dissever.runDissever(fshape, ancdts, min_iter=19, max_iter=20,
                                                              perc2evaluate=perc2evaluate,
                                                              poly2agg=admboundary2.upper(),
                                                              rastergeo=rastergeo, method=method, p=psamples,
                                                              cnnmod=cnnmodel, patchsize=patchsize, batchsize=batchsize,
                                                              epochspi=epochpi, lrate=learningrate, filters=filters,
                                                              lweights=lossweights, extdataset=extendeddataset,
                                                              yraster=yraster, converge=1.5, casestudy=casestudy,
                                                              verbose=True)

                print('- Writing raster to disk...')
                osgu.writeRaster(dissdataset, rastergeo, 'dissever_' + casestudy + '.tif')


osgu.removeShape(fshape)
