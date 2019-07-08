import ogr, gdal, osr
import numpy as np
import geopandas as gpd, pandas as pd
import os


def readRaster(file):
    raster = gdal.Open(file)
    band = raster.GetRasterBand(1)
    rastergeo = raster.GetGeoTransform()
    dataset = np.array(band.ReadAsArray())
    dataset[dataset < -999999] = np.NaN
    dataset = np.reshape(dataset, dataset.shape + (1,))
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


def addAttr2Shapefile(fshape, fcsv=None, attr=None, encoding='latin-1'):
    prj = [l.strip() for l in open(fshape.replace('.shp', '.prj'), 'r')][0]
    gdf = gpd.read_file(fshape, crs_wkt=prj, encoding=encoding)
    if attr == 'ID':
        if 'ID' not in gdf:
            print('|| Adding ID to shapefile', fshape)
            ids = list(range(1, gdf['geometry'].count() + 1))
            gdf['ID'] = ids

    else:
        print('|| Merging shapefile with csv by', attr)
        df = pd.read_csv(fcsv, sep=';', encoding=encoding)

        gdf[attr[0]] = gdf[attr[0]].str.upper()
        gdf[attr[1]] = gdf[attr[1]].str.upper()
        df[attr[0]] = df[attr[0]].str.upper()
        df[attr[1]] = df[attr[1]].str.upper()
        #print(gdf.columns)
        gdf = gdf.merge(df, left_on=attr, right_on=attr)
        #print(gdf)

    gdf.to_file(driver='ESRI Shapefile', filename=fshape, crs_wkt=prj)


def removeAttrFromShapefile(fshape, attr):
    print('|| Removing attribute(s)', attr, 'from shapefile')
    prj = [l.strip() for l in open(fshape.replace('.shp', '.prj'), 'r')][0]
    gdf = gpd.read_file(fshape, crs_wkt=prj)
    gdfattributes = list(gdf)
    for att in attr:
        if att in gdfattributes:
            gdf = gdf.drop(att, axis=1)
    gdf.to_file(driver='ESRI Shapefile', filename=fshape, crs_wkt=prj)


def removeShapefile(fshape):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    driver.DeleteDataSource(fshape)


def ogr2raster(fshape, attr, template):
    print('| Converting shapefile to raster:', fshape, '-', attr)

    if attr == 'ID':
        addAttr2Shapefile(fshape, attr='ID')

    print('|| Converting')
    source_ds = ogr.Open(fshape)
    source_layer = source_ds.GetLayer()
    # spatialRef = source_layer.GetSpatialRef()

    cols = template[1]
    rows = template[2]

    tempfile = 'tempfileo2r.tif'
    target_ds = gdal.GetDriverByName('GTiff').Create(tempfile, cols, rows, 1, gdal.GDT_Float32)

    target_ds.SetGeoTransform(template[0])
    # target_dsSRS = osr.SpatialReference()
    # target_dsSRS.ImportFromEPSG(4326)
    # target_ds.SetProjection(target_dsSRS.ExportToWkt())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(np.NaN)

    values = [row.GetField(attr) for row in source_layer]
    for i in values:
        source_layer.SetAttributeFilter(attr + '=' + str(i))
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[i])

    target_ds = None

    dataset, rastergeo = readRaster(tempfile)
    os.remove(tempfile)

    return dataset, rastergeo
