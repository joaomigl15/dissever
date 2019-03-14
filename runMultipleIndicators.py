from gith.dissever import osgeoutils as osgu, dissever
import numpy as np
import os


indicators = [['Beneficiaries17', 'nutsiii']]

ancdataset1 = osgu.readRaster('Rasters/ghspg17.tif')[0]
ancdataset2 = osgu.readRaster('Rasters/ghspg17.tif')[0]
ancdataset3 = osgu.readRaster('Rasters/ghspg17.tif')[0]
ancdataset4 = osgu.readRaster('Rasters/ghspg17.tif')[0]

ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4))
ancnames = ['LC', 'LC']


for indicator in indicators:
    print('--- Running dissever for the indicator', indicator[0])

    fshape = os.path.join('Shapefiles', (indicator[1] + 'shp'))
    fcsv = os.path.join('Statistics', (indicator[0] + 'csv'))
    yraster = 'pycno_' + indicator[0] + '.tif'

    if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, indicator[1].upper())

    tempfile = dissever.runDissever(fshape, ancdatasets, ancnames=ancnames, method='lm',
                                    patchsize = 7,
                                    yraster=yraster,
                                    max_iter=5, verbose=True)

    if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])

    print(tempfile)
