
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import scipy as sp

data = pd.read_csv('network_backup_dataset.csv') #pust it in the same working dir
data.columns = ['week', 'day_of_week', 'start_time','work_flow','file_name_str','size','time']


data['day_of_week']=data['day_of_week'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], 
                                [1,2,3,4,5,6,7])

data['work_flow']=data['work_flow'].replace(['work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4'], 
                                [0,1,2,3,4])

data['file_name_str']=data['file_name_str'].replace(['File_0','File_1','File_2','File_3','File_4','File_5','File_6','File_7','File_8','File_9',
                                             'File_10','File_11','File_12','File_13','File_14','File_15','File_16','File_17','File_18','File_19',
                                            'File_20','File_21','File_22','File_23','File_24','File_25','File_26','File_27','File_28','File_29'], 
                                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
#for i in range(0,32):
   # data['file_name']= data['file_name'].replace('File_'+str(i), i )

data.drop('time', 1, inplace = True)

linreg=linear_model.LinearRegression()
# using cross val predict function

Feature=data[['week', 'day_of_week', 'start_time','work_flow','file_name_str']]
Result=data['size']

#=========using the best parameter we found
scaler = StandardScaler()
scaler.fit(Feature)
# using best parameter, RMSE is: 0.0129735729554
    
linreg.fit(Feature,Result)
    
predicted_target = cross_val_predict(linreg, Feature, Result, cv = 10)
   
print 'After cross-validation, RMSE is: ' 
print np.sqrt(metrics.mean_squared_error(predicted_target, Result))

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


# In[ ]:



