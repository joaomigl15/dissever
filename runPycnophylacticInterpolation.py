import osgeoutils as osgu, pycno
import os


indicators = [['Belgium', 'ADMDISTRIC', 'ADMDISTRIC']]
ds, rastergeo = osgu.readRaster('Rasters/Belgium/ghspg_1975_200m.tif')
nrowsds = ds.shape[1]
ncolsds = ds.shape[0]


for indicator in indicators:
    print('--- Running pycnophylactic interpolation for the indicator', indicator[0])

    fshape = os.path.join('Shapefiles', indicator[0], (indicator[2] + '.shp'))
    fcsv = os.path.join('Statistics', indicator[0], (indicator[2] + '.csv'))


    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    osgu.addAttr2Shapefile(fshape, fcsv, [indicator[2].upper()])


    # tempfileid = None
    tempfileid = indicator[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
    pycnodataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=tempfileid)

    osgu.writeRaster(pycnodataset[:, :, 0], rastergeo, 'pycnointerpolation_' + indicator[0] + '.tif')

    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
