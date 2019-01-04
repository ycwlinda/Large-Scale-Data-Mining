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
# using cross val predict function

Feature=data[['week', 'day_of_week', 'start_time','work_flow','file_name']]
Result=data['size']

#=========using the best parameter we found

# using best parameter, RMSE is: 0.0129735729554

rf = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features=5, bootstrap=True ,oob_score=True)
    
rf.fit(Feature,Result)
    
predicted_target = cross_val_predict(rf, Feature, Result, cv = 10)
   
print 'using best parameter, RMSE is: ' 
print sp.sqrt(mean_squared_error(predicted_target, Result))

fig, ax = plt.subplots()
ax.scatter(range(0,len(Result)), Result, c = 'b', s=8, label = 'true value')
ax.scatter(range(0,len(predicted_target)), predicted_target, c = 'r' , s=8,label = 'fitted value')
ax.legend(fontsize='medium')
ax.set_xlabel('index of data points', fontsize = 20)
ax.set_ylabel('fitted and true value', fontsize = 20)
plt.title('Fitted values vs True values over all data points', fontsize = 20)

print "Fitted values vs. True values:"
plt.show()



plt.clf()
fig, ax = plt.subplots()

ax.scatter(range(0,len(Result)), predicted_target, c = 'b', s=5, label = 'fitted value')
ax.scatter(range(0,len(predicted_target)), Result- predicted_target, c = 'r' , s=5,label = 'residuals')
ax.legend(loc='upper right', fontsize='medium')
ax.set_xlabel('index of data points', fontsize = 20)
ax.set_ylabel('Residual and fitted value', fontsize = 20)
plt.title('Residuals vs Fitted value over all data points', fontsize = 20)
print "Residuals vs. Fitted value"
plt.show()
#=================================================================