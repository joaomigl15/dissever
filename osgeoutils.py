import ogr, gdal, osr
import numpy as np
import geopandas as gpd, pandas as pd
import os


def readRaster(file):
    raster = gdal.Open(file)
    band = raster.GetRasterBand(1)
    rastergeo = raster.GetGeoTransform()
    dataset = np.array(band.ReadAsArray())
    dataset[dataset < -99999999] = np.NaN
    return dataset, rastergeo


def writeRaster(dataset, rastergeo, fraster):
    driver = gdal.GetDriverByName('GTiff')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) # WGS-84
    outRaster = driver.Create(fraster, dataset.shape[1], dataset.shape[0], 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((rastergeo[0], rastergeo[1], 0, rastergeo[3], 0, rastergeo[5]))
    outRaster.SetProjection(srs.ExportToWkt())
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(-9999)
    outband.WriteArray(dataset)
    outband.FlushCache()
    outband = None


def addAttr2Shapefile(fshape, fcsv=None, attr=None):
    gdf = gpd.read_file(fshape)
    if attr == 'ID':
        print('|| Adding ID to shapefile', fshape)
        ids = list(range(1, gdf['geometry'].count() + 1))
        gdf['ID'] = ids
    else:
        print('|| Merging shapefile with csv by', attr)
        df = pd.read_csv(fcsv, sep=';')
        gdf = gdf.merge(df, on=attr)
    gdf.to_file(driver='ESRI Shapefile', filename=fshape)


def removeAttrFromShapefile(fshape, attr):
    gdf = gpd.read_file(fshape)
    print('|| Removing attribute(s)', attr, 'from shapefile')
    gdf = gdf.drop(attr, axis=1)
    gdf.to_file(driver='ESRI Shapefile', filename=fshape)


def ogr2raster(fshape, res, attr, tempfileid=None):
    print('| Converting shapefile to raster:', fshape, '-', attr)

    if attr == 'ID':
        addAttr2Shapefile(fshape, attr='ID')

    print('|| Converting')
    source_ds = ogr.Open(fshape)
    source_layer = source_ds.GetLayer()
    spatialRef = source_layer.GetSpatialRef()

    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / res)
    rows = int((y_max - y_min) / res)

    tempfile = 'tempfileo2r'
    if tempfileid: tempfile = tempfile + '_' + tempfileid
    tempfile = tempfile + '.tif'
    target_ds = gdal.GetDriverByName('GTiff').Create(tempfile, cols, rows, 1, gdal.GDT_Float32)

    target_ds.SetGeoTransform((x_min, res, 0, y_max, 0, -res))
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(4326)
    target_ds.SetProjection(target_dsSRS.ExportToWkt())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(np.NaN)

    values = [row.GetField(attr) for row in source_layer]
    for i in values:
        source_layer.SetAttributeFilter(attr + '=' + str(i))
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[i])

    target_ds = None

    dataset, rastergeo = readRaster(tempfile)
    if ~tempfileid: os.remove(tempfile)

    return dataset, rastergeo
