import geopandas as gpd
import random


def computeNeighbors(fshape):
    adjmunicipalities = {}

    gdf = gpd.read_file(fshape)

    for index, municip in gdf.iterrows():
        mname = municip.MUNICIP
        mid = municip.ID

        neigdf = gdf[~gdf.geometry.disjoint(municip.geometry)]
        for index, neig in neigdf.iterrows():
            if neig.MUNICIP != mname:
                neigid = neig.ID
                if mid in adjmunicipalities:
                    adjmunicipalities[mid].append(neigid)
                else:
                    adjmunicipalities[mid] = [neigid]

    return adjmunicipalities


def createAdjPairs(adjmunicipalities, pairperc):
    nummun = len(adjmunicipalities)
    numfinalmun = round(pairperc*nummun)
    adjpairs = []
    cpids = []

    i = 1
    while i <= numfinalmun:
        cp1 = random.randint(1, nummun)
        if (cp1 not in cpids) and (cp1 in adjmunicipalities):
            neigcp1 = adjmunicipalities[cp1]
            for cp2 in neigcp1:
                if cp2 not in cpids:
                    cpids.append(cp1)
                    cpids.append(cp2)
                    adjpairs.append([cp1, cp2])
                    i = i+1
                    break

    return adjpairs


def dissolvePairs(fshape, adjpairs):
    newpairs = {}

    i = 1
    gdf = gpd.read_file(fshape)
    for pair in adjpairs:
        gdf.loc[gdf['ID'] == pair[0], 'ID'] = int(str(99999) + str(i))
        gdf.loc[gdf['ID'] == pair[1], 'ID'] = int(str(99999) + str(i))
        newpairs[int(str(99999) + str(i))] = [pair[0], pair[1]]
        i = i + 1

    dissolveddf = gdf.dissolve(by='ID')
    dissolveddf['ID'] = dissolveddf.index

    prj = [l.strip() for l in open(fshape.replace('.shp', '.prj'), 'r')][0]
    fshape_dissolved = 'fshape_dissolved.shp'
    dissolveddf.to_file(driver='ESRI Shapefile', filename=fshape_dissolved, crs_wkt=prj)
    return fshape_dissolved, newpairs
