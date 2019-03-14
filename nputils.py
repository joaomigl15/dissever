import numpy as np


def polygonValuesByID(ds, ids):
    uniqueids = np.unique(ids[~np.isnan(ids)])

    polygonvalues = {}
    for polid in uniqueids:
        polygonvalues[polid] = ds[ids == polid][1]

    return polygonvalues


def statsByID(ds, ids, stat='sum'):
    unique, counts = np.unique(np.unique(ids[~np.isnan(ids)]), return_counts=True)
    counts = dict(zip(unique, counts))

    stats = {}
    for polid in counts:
        if stat == 'sum':
            stats[polid] = np.sum(ds[ids == polid])
        else:
            print('Invalid statistic')

    return stats
