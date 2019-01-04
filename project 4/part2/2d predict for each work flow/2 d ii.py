# working==================================================
from sklearn.metrics import mean_squared_error
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

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
poly_degrees = np.arange(1,10)

# define a cross-validation iterator
kf = KFold(n_splits=10)

algo = linear_model.LinearRegression()
all_predicted = np.zeros(data.shape[0])
all_Result = data['size']

train_RMSE_degrees = list()
test_RMSE_degrees = list()

for degree in poly_degrees:
    all_work_flow_train_RMSE = list()
    all_work_flow_test_RMSE = list()
    print 'degree = ',degree
    poly = PolynomialFeatures(degree)
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
            poly_train_FeatureSet = poly.fit_transform(train_FeatureSet)
            train_ResultSet = real_trainset['size']

            test_FeatureSet = real_testset[['week', 'day_of_week', 'start_time', 'work_flow', 'file_name']]
            poly_test_FeatureSet = poly.fit_transform(test_FeatureSet)
            test_ResultSet = real_testset['size']

            algo.fit(poly_train_FeatureSet, train_ResultSet)

            prediction_test = algo.predict(poly_test_FeatureSet)
            prediction_train = algo.predict(poly_train_FeatureSet)

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

    train_RMSE_degrees.append(np.mean(all_work_flow_train_RMSE))
    test_RMSE_degrees.append(np.mean(all_work_flow_test_RMSE))
    print 'all work flow average train RMSE', np.mean(all_work_flow_train_RMSE)
    print 'all work flow average test RMSE', np.mean(all_work_flow_test_RMSE)

plt.figure(figsize=[6,5]).set_tight_layout(True)
plt.plot(poly_degrees, train_RMSE_degrees, label='Average Train RMSE')
plt.plot(poly_degrees, test_RMSE_degrees, label='Average Test RMSE')
plt.xlabel('Degree')
plt.legend(loc="upper right")

