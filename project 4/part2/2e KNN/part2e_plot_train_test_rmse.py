# working==================================================
from sklearn.metrics import mean_squared_error
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict


data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week', 'start_time','work_flow','file_name','size','time']


data['day_of_week']=data['day_of_week'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], 
                                [1,2,3,4,5,6,7])

data['work_flow']=data['work_flow'].replace(['work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4'], 
                                [0,1,2,3,4])

for i in range(0,32):
    data['file_name']= data['file_name'].replace('File_'+str(i), i )

data.drop('time', 1, inplace = True)

#=================================================================
Feature=data[['week', 'day_of_week', 'start_time','work_flow','file_name']]
Result=data['size']

neighbors=range(1,21)

kf = KFold(n_splits=10)

train_RMSE = 0
test_RMSE =0

train_rmse=list()
test_rmse=list()

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    test_k_RMSE = list()
    train_k_RMSE = list()

    for trainset, testset in kf.split(data):
        real_trainset= data.iloc[trainset.tolist()] # from index to dataframe type
        real_testset= data.iloc[testset.tolist()]
    
        train_FeatureSet=real_trainset[['week', 'day_of_week', 'start_time','work_flow','file_name']]                                               
        train_ResultSet=real_trainset['size']

        test_FeatureSet=real_testset[['week', 'day_of_week', 'start_time','work_flow','file_name']]                                           
        test_ResultSet=real_testset['size']
    
        knn.fit(train_FeatureSet,train_ResultSet) 
        
        prediction_test = knn.predict(test_FeatureSet)
        prediction_train = knn.predict(train_FeatureSet)
        print 'working on neighbor_value = '
        print k
        test_k_RMSE.append(mean_squared_error(prediction_test, test_ResultSet))
        train_k_RMSE.append(mean_squared_error(prediction_train, train_ResultSet))


    test_RMSE = sp.sqrt(np.mean(test_k_RMSE))
    train_RMSE = sp.sqrt(np.mean(train_k_RMSE))

    train_rmse.append(train_RMSE)
    test_rmse.append(test_RMSE)

plt.figure(figsize=(10,6))
plt.plot(neighbors, train_rmse)
    
plt.xlabel('neighbors', fontsize = 20)
plt.ylabel('train_rmse ', fontsize = 20)
plt.title('train_rmse VS neighbors ', fontsize = 20)

plt.show()


plt.clf()
plt.figure(figsize = (10,6))

plt.plot(neighbors, test_rmse)

plt.xlabel('depth_value', fontsize = 20)
plt.ylabel('test_rmse', fontsize = 20)
plt.title('test_rmse VS neighbors ', fontsize = 20)

plt.show()



