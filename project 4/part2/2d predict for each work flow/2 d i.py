# working==================================================
from sklearn.metrics import mean_squared_error
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model

from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('/Volumes/DATA/Documents/Term 10/219 Project4/network_backup_dataset.csv')
data.columns = ['week', 'day_of_week', 'start_time', 'work_flow', 'file_name', 'size', 'time']

data['day_of_week'] = data['day_of_week'].replace(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    [1, 2, 3, 4, 5, 6, 7])

data['work_flow'] = data['work_flow'].replace(
    ['work_flow_0', 'work_flow_1', 'work_flow_2', 'work_flow_3', 'work_flow_4'],
    [0, 1, 2, 3, 4])

for i in range(0, 32):
    data['file_name'] = data['file_name'].replace('File_' + str(i), i)

data.drop('time', 1, inplace=True)

# =================================================================
work_flows = pd.unique(data['work_flow'])

# define a cross-validation iterator
kf = KFold(n_splits=10)

algo = linear_model.LinearRegression()
all_predicted = np.zeros(data.shape[0])
all_Result = data['size']

all_work_flow_train_RMSE = list()
all_work_flow_test_RMSE = list()

for work_flow in work_flows:
    data_this_work_flow = data.loc[data['work_flow']  == work_flow]

    Feature = data_this_work_flow[['week', 'day_of_week', 'start_time', 'work_flow', 'file_name']]
    Result = data_this_work_flow['size']

    this_RMSE = list()
    that_RMSE = list()

    for trainset, testset in kf.split(data_this_work_flow):
        real_trainset = data_this_work_flow.iloc[trainset.tolist()]  # from index to dataframe type
        real_testset = data_this_work_flow.iloc[testset.tolist()]
        train_FeatureSet = real_trainset[['week', 'day_of_week', 'start_time', 'work_flow', 'file_name']]
        train_ResultSet = real_trainset['size']

        test_FeatureSet = real_testset[['week', 'day_of_week', 'start_time', 'work_flow', 'file_name']]
        test_ResultSet = real_testset['size']

        algo.fit(train_FeatureSet, train_ResultSet)

        prediction_test = algo.predict(test_FeatureSet)
        prediction_train = algo.predict(train_FeatureSet)

        for idx, val in enumerate(prediction_test):
            all_predicted[real_testset.index[idx]] = val

        this_RMSE.append(mean_squared_error(prediction_test, test_ResultSet))
        that_RMSE.append(mean_squared_error(prediction_train, train_ResultSet))

    test_RMSE = sp.sqrt(np.mean(this_RMSE))
    train_RMSE = sp.sqrt(np.mean(that_RMSE))
    all_work_flow_test_RMSE.append(test_RMSE)
    all_work_flow_train_RMSE.append(train_RMSE)
    print 'work flow',work_flow,':'
    print 'test_RMSE',test_RMSE
    print 'train_RMSE', train_RMSE

print 'all work flow average train RMSE', np.mean(all_work_flow_train_RMSE)
print 'all work flow average test RMSE', np.mean(all_work_flow_test_RMSE)

plt.figure(figsize=[12, 12]).set_tight_layout(True)
plt.subplot(211)
plt.scatter(range(0,len(all_Result)),all_Result, c = 'b', s=1, label = 'true value', edgecolors = 'face')
plt.scatter(range(0,len(all_predicted)),all_predicted,c = 'r' , s=1, label = 'fitted value', edgecolors = 'face')
plt.legend(fontsize='medium')
plt.xlim([0.0, data.shape[0]])
plt.xlabel('index of data points', fontsize = 20)
plt.ylabel('fitted and true value', fontsize = 20)
plt.title('Fitted values vs True values over all data points', fontsize = 20)


plt.subplot(212)
plt.scatter(range(0,len(all_Result)), all_Result, c = 'b', s=1, label = 'fitted value' , edgecolors = 'face')
plt.scatter(range(0,len(all_predicted)), all_Result- all_predicted, c = 'r' , s=1,label = 'residuals' , edgecolors = 'face')
plt.legend(loc='upper right', fontsize='medium')
plt.xlim([0.0, data.shape[0]])
plt.xlabel('index of data points', fontsize = 20)
plt.ylabel('Residual and fitted value', fontsize = 20)
plt.title('Residuals vs Fitted value over all data points', fontsize = 20)