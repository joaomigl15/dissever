from gith.dissever import osgeoutils as osgu, dissever
import numpy as np
import os


indicator = 'Beneficiaries17',
admboundary = 'nutsiii'

method = ['cnn']
cnnmodel = ['lenet']
psamples = [0.25]
epochspi = [1]
batchsize = [1024]
learningrate = [0.001]
extendeddataset = [2.5]

ancnames = ['ghspg17', 'ghspg17', 'ghspg17', 'ghspg17']


fshape = os.path.join('Shapefiles', (indicator + 'shp'))
fcsv = os.path.join('Statistics', (indicator + 'csv'))
yraster = 'pycno_' + indicator + '.tif'

ancdataset1 = osgu.readRaster(os.path.join('Rasters', (ancnames[0] + '.tif')))[0]
ancdataset2 = osgu.readRaster(os.path.join('Rasters', (ancnames[1] + '.tif')))[0]
ancdataset3 = osgu.readRaster(os.path.join('Rasters', (ancnames[2] + '.tif')))[0]
ancdataset4 = osgu.readRaster(os.path.join('Rasters', (ancnames[3] + '.tif')))[0]
ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4))


if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, admboundary.upper())

for meth in method:
    if (meth != 'cnn'):
        print('--- Running dissever leveraging a', meth, 'model')
        tempfile = dissever.runDissever(fshape, ancdatasets, ancnames=ancnames, method=meth, verbose=True)
    else:
        for cnnm in cnnmodel:
            for psamp in psamples:
                for epiter in epochspi:
                    for bsize in batchsize:
                        for lrate in learningrate:
                            for edatas in extendeddataset:
                                print('--- Running dissever with the following configuration:')
                                print('- Method:', meth,
                                      '| Percentage of sampling:', psamp,
                                      '| Epochs per iteration:', epiter,
                                      '| Batch size:', bsize,
                                      '| Learning rate:', lrate,
                                      '| Extended dataset:', edatas)
                                tempfile = dissever.runDissever(fshape, ancdatasets, ancnames=ancnames,
                                                                method='cnn', cnnmodel=cnnm,
                                                                patchsize=7,
                                                                yraster=yraster,
                                                                verbose=True)

if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
