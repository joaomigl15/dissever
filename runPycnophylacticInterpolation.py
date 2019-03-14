from gith.dissever import osgeoutils as osgu, pycno
import os


indicators = [['Withdrawals18', 'nutsiii'],
              ['Purchases18', 'nutsiii']]

res = 0.00166667


for indicator in indicators:
    print('--- Running pycnophylactic interpolation for the indicator', indicator[0])

    fshape = os.path.join('Shapefiles', (indicator[1] + '.shp'))
    fcsv = os.path.join('Statistics', indicator[0], (indicator[1] + '.csv'))


    try: osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
    except: pass

    osgu.addAttr2Shapefile(fshape, fcsv, indicator[1].upper())


    idsdataset = osgu.ogr2raster(fshape, res=res, attr='ID')[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, res=res, attr='VALUE')
    disseverdataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=False)

    osgu.removeAttrFromShapefile(fshape, ['ID', 'VALUE'])
