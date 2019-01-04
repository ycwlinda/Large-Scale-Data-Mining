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
# if using cross val predict function

Feature=data[['week', 'day_of_week', 'start_time','work_flow','file_name']]
Result=data['size']

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features=5, bootstrap=True ,oob_score=True)
    
rf.fit(Feature,Result)
    
predicted_target = cross_val_predict(rf, Feature, Result, cv = 10)
   
print 'using best parameter, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, Result))


print 'rf.feature_importances_'
print rf.feature_importances_

names = ['week', 'day_of_week', 'start_time','work_flow','file_name']
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)
#=========================Evaluate importance of week==========================
print "\nIgnoring week number:"
new_data = data.copy()
new_data.drop('week', 1, inplace = True)

newFeature=new_data[[ 'day_of_week', 'start_time','work_flow','file_name']]
newResult=new_data['size']

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features=4, bootstrap=True ,oob_score=True)
    
rf.fit(newFeature,newResult)
    
predicted_target = cross_val_predict(rf, newFeature, newResult, cv = 10)

print 'rf.feature_importances_'
print(rf.feature_importances_)

print 'after ignoring week, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, newResult))

#==============================================================================


#=====================Evaluate importance of day of week=======================
print "\nIgnoring day of week:"
new_data = data.copy()
new_data.drop('day_of_week', 1, inplace = True)

newFeature=new_data[[ 'week', 'start_time','work_flow','file_name']]
newResult=new_data['size']

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features=4, bootstrap=True ,oob_score=True)
    
rf.fit(newFeature,newResult)
    
predicted_target = cross_val_predict(rf, newFeature, newResult, cv = 10)
print 'rf.feature_importances_'
print(rf.feature_importances_)

print 'after ignoring day_of_week, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, newResult))

# ==============================================================================



#=====================Evaluate importance of start time========================
print "\nIgnoring start time:"
new_data = data.copy()
new_data.drop('start_time', 1, inplace = True)

newFeature=new_data[[ 'week', 'day_of_week','work_flow','file_name']]
newResult=new_data['size']

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features=4, bootstrap=True ,oob_score=True)
    
rf.fit(newFeature,newResult)
    
predicted_target = cross_val_predict(rf, newFeature, newResult, cv = 10)
print 'rf.feature_importances_'
print(rf.feature_importances_)
   
print 'after ignoring start_time, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, newResult))

#==============================================================================



#=====================Evaluate importance of work flow=========================
print "\nIgnoring work flow:"
new_data = data.copy()
new_data.drop('work_flow', 1, inplace = True)

newFeature=new_data[[ 'week', 'day_of_week','start_time','file_name']]
newResult=new_data['size']

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9,max_features=4,  bootstrap=True ,oob_score=True)
    
rf.fit(newFeature,newResult)
    
predicted_target = cross_val_predict(rf, newFeature, newResult, cv = 10)
print 'rf.feature_importances_'
print(rf.feature_importances_)
   
print 'after ignoring work_flow, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, newResult))


#==============================================================================


#=====================Evaluate importance of file name=========================
print "\nIgnoring file name:"
new_data = data.copy()
new_data.drop('file_name', 1, inplace = True)

newFeature=new_data[[ 'week', 'day_of_week','start_time','work_flow']]
newResult=new_data['size']

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features=4, bootstrap=True ,oob_score=True)
    
rf.fit(newFeature,newResult)
    
predicted_target = cross_val_predict(rf, newFeature, newResult, cv = 10)
print 'rf.feature_importances_'
print(rf.feature_importances_)
   
print 'after ignoring file_name, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, newResult))

#==============================================================================



