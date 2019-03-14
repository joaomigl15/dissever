from gith.dissever import osgeoutils as osgu, dissever
import numpy as np


file = 'test.tif'
fwrite = 'testwrite.tif'
fshape = 'Shapefiles/admdistricts.shp'
fcsv = 'Statistics/admdistricts_1900.csv'
yraster = 'tempfilepycno.tif'
ancraster = 'Rasters/ghspg.tif'


ancdataset = osgu.readRaster(ancraster)[0]
ancdataset[ancdataset < -9999999] = 0
ancdatasets = np.dstack((ancdataset, ancdataset, ancdataset))


if not yraster: osgu.addAttr2Shapefile(fshape, fcsv, 'ADMDISTRIC')
tempfile = dissever.runDissever(fshape, ancdatasets, ancnames=['LC', 'LC'], method='cnn',
                                patchsize = 7,
                                yraster=yraster,
                                max_iter=5, verbose=True)
if not yraster: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])

print(tempfile)

