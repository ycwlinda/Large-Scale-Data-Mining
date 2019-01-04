
# coding: utf-8

# In[26]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import cross_validate
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
#import scipy as sp
from sklearn.feature_selection import f_regression
import itertools
from sklearn.preprocessing import OneHotEncoder

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


Feature=data[['week', 'day_of_week', 'start_time','work_flow','file_name_str']]
Result=data['size']
featuremat=Feature.values
resultmat=Result.values
#print featuremat

data.drop('time', 1, inplace = True)

labels=data.columns


#print featuremat
table=list(itertools.product([False,True],repeat=5))
avg_training_rmse =[]
avg_test_rmse=[]
c=0
y=resultmat
for combination in table:
    #print (str(c)+" " +str(combination))
    
    enc=OneHotEncoder(categorical_features=combination, sparse=False)
    X_enc=enc.fit_transform(featuremat)
    linreg=linear_model.LinearRegression()    
    #print X_enc
    #kf=KFold(n_splits=10,shuffle=False)
    #kf.get_n_splits(X_enc)
    index=0
    #rmse_test_average=0
    #rmse_test_average=0
    sum_rmse_test=0
    sum_rmse_train=0
    cv=cross_validate(linreg,X_enc,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True)
    train_rmse=np.mean(np.sqrt(-cv['train_score']))
    test_rmse=np.mean(np.sqrt(-cv['test_score']))
    #for trainset,testset in kf.split(X_enc):
        #real_trainset= X_enc.iloc[trainset.tolist()] # from index to dataframe type
        #real_testset= X_enc.iloc[testset.tolist()]

        #train_FeatureSet=real_trainset[['week', 'day_of_week', 'start_time','work_flow','file_name_str']]

        #train_ResultSet=real_trainset['size']

        #test_FeatureSet=real_testset[['week', 'day_of_week', 'start_time','work_flow','file_name_str']]

        #test_ResultSet=real_testset['size']

        #lr_result = linreg.fit(train_FeatureSet, train_ResultSet)

        #prediction_test = linreg.predict(test_FeatureSet)
        #prediction_train = linreg.predict(train_FeatureSet)

        #test_rmse=metrics.mean_squared_error(prediction_test, test_ResultSet)
        #train_rmse=metrics.mean_squared_error(prediction_train, train_ResultSet)
        #sum_rmse_test += test_rmse
        #sum_rmse_train +=train_rmse
        #index+=1
    #rmse_test_average=sum_rmse_test/10
    #rmse_train_average=sum_rmse_train/10
    avg_training_rmse.append(train_rmse)
    avg_test_rmse.append(test_rmse)
    
    binary_c="{0:05b}".format(c)
    print binary_c
    c=c+1
    print "Train RMSE = %.7f, test RMSE =%.7f" % (train_rmse,test_rmse)    
#print avg_training_rmse, avg_test_rmse

fig,ax=plt.subplots()
ax.plot(avg_training_rmse,label="Average Training RMSE")
ax.set_xlabel('combination')
ax.set_ylabel('Training RMSE')
plt.show()

fig,ax=plt.subplots()
ax.plot(avg_test_rmse,label="Average Test RMSE")
ax.set_xlabel('combination')
ax.set_ylabel('Test RMSE')
plt.show()


# In[50]:

def findbestcomb(clf, X, y, verbose=True):
    train_rmse_lst = []
    test_rmse_lst = []

    tftable = list(itertools.product([False, True], repeat=5))
    for comb in tftable:
        enc = OneHotEncoder(categorical_features=comb, sparse=False)
        X_enc = enc.fit_transform(np.float64(X))
        cv_results = cross_validate(clf, X_enc, y,                 scoring='neg_mean_squared_error', cv=10, return_train_score=True)
        train_rmse = np.mean(np.sqrt(-cv_results['train_score']))
        test_rmse = np.mean(np.sqrt(-cv_results['test_score']))
        train_rmse_lst.append(train_rmse)
        test_rmse_lst.append(test_rmse)

    best_comb_idx = np.argmin(test_rmse_lst)
    if verbose:
        print("Best combination is: ", tftable[best_comb_idx])
        #print([df_index[i] for i, x in enumerate(tftable[best_comb_idx]) if x])
        print("Train RMSE, Test RMSE: ", train_rmse_lst[best_comb_idx], test_rmse_lst[best_comb_idx])
    
    return (tftable[best_comb_idx], best_comb_idx, train_rmse_lst, test_rmse_lst)


(_, best_comb_idx, train_rmse_lst, test_rmse_lst) = findbestcomb(linreg, featuremat, y)

from sklearn.linear_model import Ridge
tftable = list(itertools.product([False, True], repeat=5))
min_rmse_r = 1
min_a_r = 0
min_comb_r = tftable[0]
for a in [5**num for num in range(-7,8)]:
    ridgeregression = Ridge(alpha=a)
    #print a
    (bc, bci, trn, tst) = findbestcomb(ridgeregression, featuremat, y, verbose=False)
    if tst[bci] < min_rmse_r:
        min_a_r = a
        min_comb_r = bc
        min_rmse_r = tst[bci]
        
print("With ridge regression, best alpha \nand best combination is: ", (min_a_r, min_comb_r))
#print([df_index[i] for i, x in enumerate(min_comb_r) if x])
print("Test RMSE: ", min_rmse_r)


# In[31]:

print listcomb[np.argsort(listerror)[0]],listerror[np.argsort(listerror)[0]]


# In[47]:

from sklearn.linear_model import Lasso

min_rmse_l = 1
min_a_l = 0
min_comb_l = tftable[0]
for a in [5**num for num in range(-7,8)]:
    lassoregression = Lasso(alpha=a)
    (bc, bci, trn, tst) = findbestcomb(lassoregression, featuremat, y, verbose=False)
    if tst[bci] < min_rmse_l:
        min_a_l = a
        min_comb_l = bc
        min_rmse_l = tst[bci]
        
print("With Lasso regression, best alpha \nand best combination is: ", (min_a_l, min_comb_l))
#print([df_index[i] for i, x in enumerate(min_comb_l) if x])
print("Test RMSE: ", min_rmse_l)


# In[ ]:



