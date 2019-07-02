import osgeoutils as osgu, pycno
import os


indicators = [['Withdrawals18', 'nutsiii'], ['Purchases18', 'nutsiii']]
ds, rastergeo = osgu.readRaster('Rasters/ghs15.tif')
nrowsds = ds.shape[1]
ncolsds = ds.shape[0]


for indicator in indicators:
    print('--- Running pycnophylactic interpolation for the indicator', indicator[0])

    fshape = os.path.join('Shapefiles', (indicator[1] + '.shp'))
    fcsv = os.path.join('Statistics', indicator[0], (indicator[1] + '.csv'))


    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    osgu.addAttr2Shapefile(fshape, fcsv, indicator[1].upper())


    tempfileid = None #indicator[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
    pycnodataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=tempfileid)

    osgu.writeRaster(pycnodataset, rastergeo, 'pycnointerpolation_' + indicator[0] + '.tif')

    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
