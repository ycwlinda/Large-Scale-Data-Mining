# working==================================================
from sklearn.metrics import mean_squared_error
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split


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

RMSE=list()
OOBerror=list()
plot_limit=200
RMSE = {0:[0.0]*plot_limit, 1:[0.0]*plot_limit, 2:[0.0]*plot_limit, 3:[0.0]*plot_limit, 4:[0.0]*plot_limit}
OOBerror = {0:[0.0]*plot_limit, 1:[0.0]*plot_limit, 2:[0.0]*plot_limit, 3:[0.0]*plot_limit, 4:[0.0]*plot_limit}


tree_value=range(1,201)
for num_tree in tree_value:
    this_RMSE = list()
    OutOfBagError=list()
    for num_feature in range(1,6):
        rf = RandomForestRegressor(n_estimators = num_tree, max_depth = 4, max_features=num_feature, bootstrap=True ,oob_score=True)
        rf.fit(Feature,Result)
        predicted_target = cross_val_predict(rf, Feature, Result, cv = 10)
        print 'working on tree_value = '
        print num_tree
        this_RMSE.append(sp.sqrt(mean_squared_error(predicted_target, Result)))
        
        oob_error = 1 - rf.oob_score_
        OutOfBagError.append(oob_error)
    
        OOBerror[num_feature-1][num_tree-1] = OutOfBagError[num_feature-1] 
    
        RMSE[num_feature-1][num_tree-1] = this_RMSE[num_feature-1] 
        
color_list = ['b', 'g', 'r', 'c', 'm']
    
plt.figure(figsize=(10,6))
for i in range(0, 5):
    plt.plot(tree_value, RMSE[i], c = color_list[i],  label = 'Feature '+str(i+1))
    
plt.xlabel('tree_value', fontsize = 20)
plt.ylabel('Test RMSE ', fontsize = 20)
plt.title('Test RMSE VS num of trees ', fontsize = 20)
plt.legend(loc="upper right")
plt.show()

# =================================================
plt.clf()
plt.figure(figsize = (10,6))
for i in range(0, 5):
    plt.plot(tree_value, OOBerror[i], c = color_list[i],  label = 'Feature '+str(i+1))

plt.xlabel('tree_value', fontsize = 20)
plt.ylabel('OOBerror', fontsize = 20)
plt.title('OOBerror VS num of trees ', fontsize = 20)
plt.legend(loc="upper right")
plt.show()


