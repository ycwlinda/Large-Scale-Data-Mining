from sklearn.metrics import mean_squared_error
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz 
from sklearn import tree
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

rf = RandomForestRegressor(n_estimators = 12, max_depth = 4, max_features=5, bootstrap=True ,oob_score=True)
    
rf.fit(Feature,Result)
    
predicted_target = cross_val_predict(rf, Feature, Result, cv = 10)
   
print 'using best parameter, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, Result))

dot_data = tree.export_graphviz(rf.estimators_[1], out_file=None,
                        feature_names=['week', 'day_of_week', 'start_time','work_flow','file_name'],  
                         class_names=['size'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  

# pydotplus.graph_from_dot_data(graph.getvalue()).write_png('dtree_1.png')

graph.view()

