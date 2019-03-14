import os
from qgis.analysis import QgsZonalStatistics


indicators = [['Withdrawals18', 'municip'],
              ['Purchases18', 'municip']]


for indicator in indicators:
    print('--- Computing statistics for the indicator ' + indicator[0])

    shape_portugal = QgsVectorLayer('/mnt/joao/Pycharm/dissever/Shapefiles/' + indicator[1] + '.shp', 'zonepolygons', 'ogr')

    for name_dir in os.listdir('/mnt/joao/Pycharm/dissever/Results'):
        dir_2evaluate = os.path.join('/mnt/joao/Pycharm/dissever/Results', name_dir, '2Evaluate')
        if os.path.isdir(dir_2evaluate):
            for name_file in os.listdir(dir_2evaluate):
                file_2evaluate = os.path.join(dir_2evaluate, name_file)
                if os.path.isfile(file_2evaluate):
                    raster = QgsRasterLayer(file_2evaluate)
                    zoneStat = QgsZonalStatistics (shape_portugal, raster, '', 1, QgsZonalStatistics.Sum)
                    zoneStat.calculateStatistics(None)
                    idx = shape_portugal.dataProvider().fieldNameIndex('sum')
                    shape_portugal.startEditing()
                    shape_portugal.renameAttribute(idx, 'VALUE')
                    shape_portugal.commitChanges()
                    createopts=['SEPARATOR=SEMICOLON']
                    name_file = name_file.replace('.tif', '')
                    name_out = os.path.join('/mnt/joao/Pycharm/dissever/Estimates', name_dir, '2Evaluate', name_file) + '.csv'
                    QgsVectorFileWriter.writeAsVectorFormat(shape_portugal, name_out, 'utf-8', driverName='CSV', layerOptions=createopts)
                    shape_portugal.dataProvider().deleteAttributes([idx])
                    shape_portugal.updateFields()

                    destination = str('/mnt/joao/Pycharm/dissever/Results/' +  name_dir + '/' + name_file + '.tif')
                    os.rename(file_2evaluate, destination)


#QgsProject.instance().addMapLayer(shape)