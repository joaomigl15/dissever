import osgeoutils as osgu, kerasutils as ku, caret
import pycno


def runDissever(fshape, ancdatasets, ancnames, yraster=None, method='lm',
                cnnmodel='lenet', patchsize = 7, epochspi=1, batchsize=1024,
                min_iter=5, max_iter=100, p=None,
                verbose=False):

    print('| DISSEVER')
    if yraster:
        disseverdataset, rastergeo = osgu.readRaster(yraster)
    else:
        fpycnoraster = pycno.runPycno(fshape, niter=5, res=0.00166667)
        disseverdataset, rastergeo = osgu.readRaster(fpycnoraster)

    if(method == 'cnn'):
        cnnmodel = ku.compilecnnmodel(cnnmodel)
        ancpatches = ku.createpatches(ancdatasets, patchsize)
        disspatches = ku.createpatches(disseverdataset, patchsize)

    for k in range(1, max_iter+1):
        if(verbose): print('| - Iteration', k)

        if(method != 'cnn'):
            mod = caret.fit(ancdatasets, disseverdataset, method)
        else:
            mod = caret.fitcnn(ancpatches, disspatches, cnnmodel, epochs=epochspi, batchsize=batchsize)

        disseverdataset = caret.predict(mod, ancdatasets)

        #if(verbose): print('| -- Computing adjustement factor')

    tempfile = 'tempfiledissever.tif'
    osgu.writeRaster(disseverdataset, rastergeo, tempfile)
    return tempfile
