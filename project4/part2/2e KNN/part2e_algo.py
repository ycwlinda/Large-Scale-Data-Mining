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
from sklearn.ensemble import RandomForestRegressor

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

RMSE1=list()
RMSE2=list()
RMSE3=list()
RMSE4=list()
neighbors=range(1,201)
for k in neighbors:
    knn1= KNeighborsRegressor(n_neighbors=k,weights='distance',algorithm ='auto')
#     algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    knn2= KNeighborsRegressor(n_neighbors=k,weights='distance',algorithm ='ball_tree')
    knn3= KNeighborsRegressor(n_neighbors=k,weights='distance',algorithm ='kd_tree')
    knn4= KNeighborsRegressor(n_neighbors=k,weights='distance',algorithm ='brute')
    knn1.fit(Feature,Result)
    knn2.fit(Feature,Result)
    knn3.fit(Feature,Result)
    knn4.fit(Feature,Result)
    predicted_target1 = cross_val_predict(knn1, Feature, Result, cv = 10)
    predicted_target2 = cross_val_predict(knn2, Feature, Result, cv = 10)
    predicted_target3 = cross_val_predict(knn3, Feature, Result, cv = 10)
    predicted_target4 = cross_val_predict(knn4, Feature, Result, cv = 10)
    print 'working on neighbor_value = '
    print k

    RMSE1.append(sp.sqrt(mean_squared_error(predicted_target1, Result)))
    RMSE2.append(sp.sqrt(mean_squared_error(predicted_target2, Result)))
    RMSE3.append(sp.sqrt(mean_squared_error(predicted_target3, Result)))
    RMSE4.append(sp.sqrt(mean_squared_error(predicted_target4, Result)))
plt.figure(figsize=(10,6))
plt.plot(neighbors, RMSE1 , c = 'b',label = 'auto')
plt.plot(neighbors, RMSE2 , c = 'r',label = 'ball_tree')
plt.plot(neighbors, RMSE3 , c = 'g',label = 'kd_tree')
plt.plot(neighbors, RMSE4 , c = 'c',label = 'brute')
plt.legend(loc='upper right', fontsize='medium')
plt.xlabel('neighbor_value', fontsize = 20)
plt.ylabel('Test RMSE ', fontsize = 20)
plt.title('RMSE VS num of neighbors using several algorithm', fontsize = 20)

plt.show()