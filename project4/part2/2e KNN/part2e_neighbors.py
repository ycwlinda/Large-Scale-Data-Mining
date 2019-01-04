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

this_RMSE=list()
neighbors=range(1,201)
for k in neighbors:
    knn= KNeighborsRegressor(n_neighbors=k)
    knn.fit(Feature,Result)
    predicted_target = cross_val_predict(knn, Feature, Result, cv = 10)
    print 'working on neighbor_value = '
    print k

    this_RMSE.append(sp.sqrt(mean_squared_error(predicted_target, Result)))


plt.figure(figsize=(10,6))
plt.plot(neighbors, this_RMSE )
    
plt.xlabel('neighbor_value', fontsize = 20)
plt.ylabel('Test RMSE ', fontsize = 20)
plt.title('Test RMSE VS num of neighbors ', fontsize = 20)

plt.show()
        