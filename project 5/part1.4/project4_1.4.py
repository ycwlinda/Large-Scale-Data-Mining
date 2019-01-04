
# coding: utf-8

# In[13]:


import json
import matplotlib.pyplot as plt
import datetime, time
import pytz
import pandas as pd


# In[14]:


import os
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import math
path='E:/ee219/tweet_data/' #path of the dataset
filelist=os.listdir(path)
filepathlist=[]
for files in filelist:
    filepath=os.path.join(path,files)
    filepathlist.append(filepath)
print filepathlist


# In[15]:


import datetime, time
import pytz

start_time = time.mktime(time.strptime("2015-02-01 08:00:00",'%Y-%m-%d %H:%M:%S'))
end_time = time.mktime(time.strptime("2015-02-01 20:00:00",'%Y-%m-%d %H:%M:%S'))
# used as the zero point of the time
base_time = time.mktime(time.strptime("2015-01-01 00:00:00",'%Y-%m-%d %H:%M:%S'))
print start_time,end_time,base_time  


# In[16]:


kf = KFold(n_splits = 10, shuffle = True)
#perform k-fold  10
def k_fold(X, y, model):
   
    MSE=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #print(sm.OLS(y, X).fit().summary())
        model.fit(X_train,y_train)
        predict = model.predict(X_test)
        
        error=mean_squared_error(y_test,predict)
        MSE.append(error)
        #print("Errors from each folder is: ", error)
    #print("Averaged error is: ", np.mean(MSE))
        rmse=math.sqrt(np.mean(MSE))
    return  rmse


# In[17]:



classifiers = [
    SVC(),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    LinearRegression(fit_intercept = False)]

hashtag_set=['#GoHawks','#GoPatriots','#NFL','#Patriots','#SB49','#SuperBowl']
hashtag_dict = {'#GoHawks': 'tweets_#gohawks.txt',
                    '#GoPatriots': 'tweets_#gopatriots.txt',
                    '#NFL': 'tweets_#nfl.txt',
                    '#Patriots': 'tweets_#patriots.txt',
                    '#SB49': 'tweets_#sb49.txt',
                    '#test': 'tweets_#test.txt',
                    '#SuperBowl': 'tweets_#superbowl.txt'}
for clf in classifiers:
    print clf
    for hashtag in hashtag_set:
        time_stamps=[]
        input_file = open(path+ hashtag_dict[hashtag])
        for line in input_file:
            data = json.loads(line)
            time_stamps.append(data['citation_date'])
        #print min(time_stamps)    
        new_starttime=int((start_time-min(time_stamps))/3600)+1
        new_endtime=int((end_time-min(time_stamps))/3600)+1

        #print new_starttime,new_endtime

        #for hashtag in hashtag_set:
        data = pd.read_csv('./Extracted_data/Q3_'+hashtag+'.csv')

        df=data[['time_windows','num_tweets','num_authors','ranking_score','target']]
        firstdf=df[df.time_windows<new_starttime]
        seconddf=df[(df.time_windows>new_starttime)&(df.time_windows<new_endtime)]
        thirddf=df[df.time_windows>new_endtime]

        firsttarget=firstdf.pop('target')

        secondtarget=seconddf.pop('target')
        thirdtarget=thirddf.pop('target')

        firstperiod=firstdf[['num_tweets','num_authors','ranking_score']]
        secondperiod=seconddf[['num_tweets','num_authors','ranking_score']]
        thirdperiod=thirddf[['num_tweets','num_authors','ranking_score']]

        firstmat=firstperiod.as_matrix()
        firstlabel=firsttarget.as_matrix()
        secondmat=secondperiod.as_matrix()
        secondlabel=secondtarget.as_matrix()
        thirdmat=thirdperiod.as_matrix()
        thirdlabel=thirdtarget.as_matrix()
        #print np.size(firstmat)
        #print np.size(firstlabel)
    
   
        
        print hashtag
        print ':Before Feb 01 8am',k_fold(firstmat,firstlabel,clf)

        print ':Between Feb 01 8am and 8pm',k_fold(secondmat,secondlabel,clf)

        print ':After Feb 01 8pm',k_fold(thirdmat,thirdlabel,clf)


# In[18]:


#aggregate all 6 data together


# In[26]:


df_total=[]

classifiers = [
    SVC(),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    LinearRegression(fit_intercept = False)]
for clf in classifiers:
    
    for hashtag in hashtag_set:
        data = pd.read_csv('./Extracted_data/Q3_'+hashtag+'.csv') 
        df_total.append(data)


    dframe=pd.concat(df_total)
    df=dframe[['time_windows','num_tweets','num_authors','ranking_score','target']]
    firstdf=df[df.time_windows<new_starttime]
    seconddf=df[(df.time_windows>new_starttime)&(df.time_windows<new_endtime)]
    thirddf=df[df.time_windows>new_endtime]
    firstmat=firstperiod.as_matrix()
    firstlabel=firsttarget.as_matrix()
    secondmat=secondperiod.as_matrix()
    secondlabel=secondtarget.as_matrix()
    thirdmat=thirdperiod.as_matrix()
    thirdlabel=thirdtarget.as_matrix()
    print clf
    print 'Before Feb 01 8am',k_fold(firstmat,firstlabel,clf)

    print 'Between Feb 01 8am and 8pm',k_fold(secondmat,secondlabel,clf)

    print 'After Feb 01 8pm',k_fold(thirdmat,thirdlabel,clf)

