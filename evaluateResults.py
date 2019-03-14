from sklearn import metrics
import numpy as np, pandas as pd
import os, shutil


indicators = [['Beneficiaries17', 'NUTSIII', 'MUNICIPALITY']]


def rmse_error(actual, predicted):
    return np.sqrt(metrics.mean_squared_error(actual, predicted))

def mae_error(actual, predicted):
    return metrics.mean_absolute_error(actual, predicted)

def nrmse_error(actual, predicted):
    range = max(actual) - min(actual)
    return (np.sqrt(metrics.mean_squared_error(actual, predicted)))/range

def nmae_error(actual, predicted):
    range = max(actual) - min(actual)
    return (metrics.mean_absolute_error(actual, predicted))/range


for indicator in indicators:
    print('- Evaluating', indicator[0])

    # Read census .csv
    data_census = pd.read_csv(os.path.join('Statistics', indicator[0], indicator[2]))

    path = os.path.join('Estimates', indicator[0], '2Evaluate')
    newpath = os.path.join('Estimates', indicator[0])

    file_names = os.listdir(path)
    for file in file_names:
        if file.endswith('.csv'):
            data_estimated = pd.read_csv(file, sep=";")

            actual = []
            predicted = []
            for index, row in data_estimated.iterrows():
                name_ab1 = row[indicator[1]]
                name_ab2 = row[indicator[2]]
                predicted_value = row['VALUE']

                actual_value = data_census.loc[(data_census[indicator[1]] == name_ab1) &
                                               (data_census[indicator[2]] == name_ab2), 'VALUE']

                if (len(actual_value.index) == 1):
                    actual.append(actual_value)
                    predicted.append(predicted_value)

            r1 = round(rmse_error(actual, predicted), 1)
            r2 = round(mae_error(actual, predicted), 1)
            r3 = round(nrmse_error(actual, predicted), 4)
            r4 = round(nmae_error(actual, predicted), 4)

            print('-- Error metrics:')
            print('--', r1, '&', r2, '&', r3, '&', r4)

            shutil.copyfile(file, newpath)
            os.remove(file)
