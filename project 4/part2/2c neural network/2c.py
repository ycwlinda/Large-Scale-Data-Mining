
# coding: utf-8

# In[2]:


import pandas as pd

network_backups = pd.read_csv('network_backup_dataset.csv')
size_column = network_backups['Size of Backup (GB)']
work_flow_column = network_backups['Work-Flow-ID']
week_column = network_backups['Week #']
day_of_week_column = network_backups['Day of Week']
start_time_column = network_backups['Backup Start Time - Hour of Day']
file_name_column = network_backups['File Name']

# Add day column
map_day = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
day = []
for i in range(0, 18588):
    day.append((week_column[i]-1)*7+map_day[day_of_week_column[i]])
network_backups.insert(2, 'day', day)

#Preprocessing day of week 
day_of_week_dic = {'Monday':[1,0,0,0,0,0,0], 
                   'Tuesday':[0,1,0,0,0,0,0],
                   'Wednesday':[0,0,1,0,0,0,0],
                   'Thursday':[0,0,0,1,0,0,0],
                   'Friday':[0,0,0,0,1,0,0],
                   'Saturday':[0,0,0,0,0,1,0],
                   'Sunday':[0,0,0,0,0,0,1]}
day_of_week_one_hot = []

for item in day_of_week_column:
    day_of_week_one_hot.append(day_of_week_dic[item])

#Preprocessing week 

from numpy import array
from sklearn.preprocessing import OneHotEncoder

week = array([i for i in range(1, 16)])
onehot_encoder = OneHotEncoder(sparse=False).fit(week.reshape(len(week), 1))
week_encoded = onehot_encoder.transform(week_column.values.reshape(len(week_column),1)).tolist()

#Preprocessing start time 
start_time_encoded = onehot_encoder.fit_transform(start_time_column.values.reshape(len(start_time_column), 1))

#Preprocessing work flow 
work_flow_encoded = []
work_flow = {'work_flow_0':0, 'work_flow_1':1, 'work_flow_2':2, 'work_flow_3':3, 'work_flow_4':4}
onehot_encoder = OneHotEncoder(sparse=False).fit(array([i for i in work_flow.values()]).reshape(5, 1))
for i in work_flow_column:
    work_flow_encoded.append(*onehot_encoder.transform(work_flow[i]).tolist())

#Preprocessing file name 
file_name_encoded = []
file_name_dic = {}
for i in file_name_column:
    file_name_dic[i] = int(i[5:])

onehot_encoder = OneHotEncoder(sparse=False).fit(array([i for i in file_name_dic.values()]).reshape(len(file_name_dic), 1))
for i in file_name_column:
    file_name_encoded.append(*onehot_encoder.transform(file_name_dic[i]).tolist())
    
import numpy as np
theinput = np.hstack((day_of_week_one_hot, start_time_encoded, work_flow_encoded, file_name_encoded, week_encoded))    

np.shape(theinput)


# In[3]:


import math
from sklearn.metrics import mean_squared_error

def calculate_RMSE(predicted, actual):  
    return mean_squared_error(actual, predicted)


# In[4]:


#from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import math
from math import sqrt
train_data = theinput.copy()
target = size_column
kf = KFold(n_splits=10)
# relu
relu = []
for i in range(50, 550, 50):
   # print('Number of hidden units = ' + str(i))
    train_RMSE =[]
    test_RMSE = []
    neu_net_reg = MLPRegressor(hidden_layer_sizes=(i), max_iter=200)
    
    for train_index, test_index in kf.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = target[train_index], target[test_index]
    
        neu_net_reg.fit(X_train, y_train)

        train_RMSE.append(calculate_RMSE(y_train, neu_net_reg.predict(X_train)))
        test_RMSE.append(calculate_RMSE(y_test, neu_net_reg.predict(X_test)))
    
    relu.append(str(math.sqrt(sum(test_RMSE)/len(test_RMSE))))
    #print('  Average train RMSE = '+ str(math.sqrt(sum(train_RMSE)/len(train_RMSE))))
   # print('  Average test RMSE = '+ str(math.sqrt(sum(test_RMSE)/len(test_RMSE))))


# In[18]:


# logistic
logistic = []
for i in range(50, 550, 50):
    #print('Number of hidden units = ' + str(i))
    train_RMSE =[]
    test_RMSE = []
    neu_net_reg = MLPRegressor(hidden_layer_sizes=(i), activation = 'logistic', max_iter=200)
    
    for train_index, test_index in kf.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = target[train_index], target[test_index]
    
        neu_net_reg.fit(X_train, y_train)

        train_RMSE.append(calculate_RMSE(y_train, neu_net_reg.predict(X_train)))
        test_RMSE.append(calculate_RMSE(y_test, neu_net_reg.predict(X_test)))
    
    logistic.append(str(math.sqrt(sum(test_RMSE)/len(test_RMSE)))) 
    


# In[19]:


# tanh
tanh = []
for i in range(50, 550, 50):
    print('Number of hidden units = ' + str(i))
    train_RMSE =[]
    test_RMSE = []
    neu_net_reg = MLPRegressor(hidden_layer_sizes=(i), activation = 'tanh', max_iter=200)
    
    for train_index, test_index in kf.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = target[train_index], target[test_index]
    
        neu_net_reg.fit(X_train, y_train)

        train_RMSE.append(calculate_RMSE(y_train, neu_net_reg.predict(X_train)))
        test_RMSE.append(calculate_RMSE(y_test, neu_net_reg.predict(X_test)))
    
    tanh.append(str(math.sqrt(sum(test_RMSE)/len(test_RMSE)))) 
    print('  Average train RMSE = '+ str(math.sqrt(sum(train_RMSE)/len(train_RMSE))))
    print('  Average test RMSE = '+ str(math.sqrt(sum(test_RMSE)/len(test_RMSE))))


# In[33]:


import matplotlib.pyplot as plt
fig,ax = plt.subplots()

x = np.arange(50, 550, 50)
#x=np.linspace(50, 550, 10)
ax.plot(x, relu, label='Relu')
ax.plot(x, logistic, label='Logistic')
ax.plot(x, tanh, label='Tanh')
ax.set_xlabel('Number of Hidden Nodes')
ax.set_ylabel('Test RMSE')
plt.legend()
plt.show()


# In[34]:


from sklearn.model_selection import train_test_split

neu_net_reg = MLPRegressor(hidden_layer_sizes=(550), max_iter=200)

for train_index, test_index in kf.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    neu_net_reg.fit(X_train, y_train)

plt.figure(figsize = (15,7.5))
plt.scatter(range(1, len(target)+1), neu_net_reg.predict(train_data), c='red',s=5)
plt.scatter(range(1, len(target)+1), target, c='blue',s=5)
plt.title('Fitted values vs True values over all data points', fontsize = 20)

print "Fitted values vs. True values:"
plt.show()


# In[35]:


plt.figure(figsize = (15,7.5))
plt.scatter(range(1, len(target)+1),neu_net_reg.predict(train_data), c='red',s=5)
plt.scatter(range(1, len(target)+1), target-neu_net_reg.predict(train_data), c='blue',s=5)
ax.set_xlabel('index of data points', fontsize = 20)
ax.set_ylabel('Residual and fitted value', fontsize = 20)
plt.title('Residuals vs Fitted value over all data points', fontsize = 20)
print "Residuals vs. Fitted value"
plt.show()

