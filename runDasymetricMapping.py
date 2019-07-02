import osgeoutils as osgu, dasymmapping as dm
import os


indicators = [['LiveBirths18', 'nutsiii', 'municip']]
ds, rastergeo = osgu.readRaster('Rasters/Normalized/ghs15_200m.tif')
nrowsds = ds.shape[1]
ncolsds = ds.shape[0]


for indicator in indicators:
    print('--- Running dasymetric mapping for the indicator', indicator[0])

    fshape = os.path.join('Shapefiles', (indicator[2] + '13_cont.shp'))
    fcsv = os.path.join('Statistics', indicator[0], (indicator[2] + '.csv'))

    fancdataset = 'Rasters/Normalized/ghs15_200m.tif'


    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    osgu.addAttr2Shapefile(fshape, fcsv, [indicator[1].upper(), indicator[2].upper()], encoding='UTF-8')

    tempfileid = None #None
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
    ancdataset = osgu.readRaster(fancdataset)[0]
    tddataset, rastergeo = dm.rundasymmapping(idsdataset, polygonvaluesdataset, ancdataset, rastergeo, tempfileid=tempfileid)

    tddataset[tddataset < 0] = 0
    tddataset = tddataset[:,:,0]
    osgu.writeRaster(tddataset, rastergeo, 'td_' + indicator[0] + '.tif')

    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
