import osgeoutils as osgu, dasymmapping as dm
import os


indicators = [['Britain', 'regdistric', 'regdistric']]
ds, rastergeo = osgu.readRaster('Rasters/Britain/ghs15_t_200m.tif')
nrowsds = ds.shape[1]
ncolsds = ds.shape[0]


for indicator in indicators:
    print('--- Running dasymetric mapping for the indicator', indicator[0])

    fshape = os.path.join('Shapefiles/Britain', (indicator[2] + '.shp'))
    fcsv = os.path.join('Statistics', indicator[0], (indicator[2] + '.csv'))

    fancdataset = 'Rasters/Britain/ghs15_t_200m.tif'


    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    # osgu.addAttr2Shapefile(fshape, fcsv, [indicator[1].upper(), indicator[2].upper()], encoding='UTF-8')
    osgu.addAttr2Shapefile(fshape, fcsv, [indicator[2].upper()], encoding='UTF-8')

    tempfileid = None #None
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
    ancdataset = osgu.readRaster(fancdataset)[0]
    tddataset, rastergeo = dm.rundasymmapping(idsdataset, polygonvaluesdataset, ancdataset, rastergeo, tempfileid=tempfileid)

    #tddataset[tddataset < 0] = 0
    osgu.writeRaster(tddataset[:,:,0], rastergeo, 'td_' + indicator[0] + '.tif')

    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
