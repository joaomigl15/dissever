import numpy as np


def polygonValuesByID(ds, ids):
    uniqueids = np.unique(ids)

    polygonvalues = {}
    for polid in uniqueids:
        if polid != -9999:
            polygonvalues[polid] = ds[ids == polid][1]

    return polygonvalues


def statsByID(ds, ids, stat='sum'):
    unique, counts = np.unique(ids, return_counts=True)
    counts = dict(zip(unique, counts))

    stats = {}
    for polid in counts:
        if polid != -9999:
            if stat == 'sum':
                stats[polid] = np.sum(ds[ids == polid])
            else:
                print('Invalid statistic')

    return stats
