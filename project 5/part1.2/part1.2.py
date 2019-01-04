"""
PROBLEM 2 Objective
Fit a linear regression model with 5 features to predict # of tweets in the next
hour, from data in the PREVIOUS hour.
Fetures:
1. number of tweets
2. number of retweets
3. sum of followers of the users posting the hashtag
4. max numbber of followers of the user posting the hashtag
5. time of the dat (24 hours represent the day)
Explain your model's training accuracy and the significance of each feature using
the t-test and P-value results of fitting the model.
"""

import os
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#===================== Extract data from file ===================== 

hashtag_set=['#GoHawks','#GoPatriots','#NFL','#Patriots','#SB49','#SuperBowl']
for hashtag in hashtag_set:
    hashtag_dict = {'#GoHawks': 'tweets_#gohawks.txt',
                    '#GoPatriots': 'tweets_#gopatriots.txt',
                    '#NFL': 'tweets_#nfl.txt',
                    '#Patriots': 'tweets_#patriots.txt',
                    '#SB49': 'tweets_#sb49.txt',
                    '#test': 'tweets_#test.txt',
                    '#SuperBowl': 'tweets_#superbowl.txt'}
    time_stamps = []
    followers_of_users = []
    is_retweet = []
    num_of_retweets = []

    input_file = open('/Users/yucongwang/Documents/EE219/project5/tweet_data/' + hashtag_dict[hashtag])
    for line in input_file:
        data = json.loads(line)
        time_stamps.append(data['citation_date'])
        followers_of_users.append(data['author']['followers'])
        
        author_name = data['author']['nick']
        original_author_name = data['original_author']['nick']
        if author_name != original_author_name:
            is_retweet.append(True)
        else:
            is_retweet.append(False)
            
    input_file.close()

 #===================== Calculate parameters based on hour index =====================

    total_hours = int((max(time_stamps)-min(time_stamps))/3600)+1
    hourly_number_of_tweets = [0] * total_hours
    hourly_number_of_retweets = [0] * total_hours
    hourly_sum_of_followers = [0] * total_hours
    hourly_max_number_of_followers = [0] * total_hours
    hourly_time_of_the_day = [0] * total_hours
    time_windows = [0] * total_hours
    start_time = min(time_stamps)
    
    for i in range(0, len(time_stamps)):
        current_hour = int((time_stamps[i]-start_time)/3600)
        
        hourly_number_of_tweets[current_hour] += 1
        if is_retweet[i]:
            hourly_number_of_retweets[current_hour] += 1
                                      
        hourly_sum_of_followers[current_hour] += followers_of_users[i]
    
        if followers_of_users[i] > hourly_max_number_of_followers[current_hour]:
            hourly_max_number_of_followers[current_hour] = followers_of_users[i]
    
    for i in range(0, total_hours):
        time_windows[i] = i
        hourly_time_of_the_day[i] = i%24

    
#===================== Build DataFrame and save to file =====================

    target_value = hourly_number_of_tweets[1:]
    target_value.append(0)
    data = np.array([
                time_windows,
                hourly_time_of_the_day,
                hourly_number_of_tweets,
                hourly_number_of_retweets,
                hourly_sum_of_followers,
                hourly_max_number_of_followers,
                target_value])
    data = np.transpose(data)
    df = DataFrame(data)
    df.columns = [
                  'time_windows',
                  'time_of_the_day',
                  'num_tweets', 
                  'num_retweets', 
                  'sum_followers',
                  'max_followers',
                  'target_value']
    if os.path.isdir('./Extracted_data'):
        pass
    else:
        os.mkdir('./Extracted_data')
    df.to_csv('./Extracted_data/Q2_'+hashtag+'.csv', index = False)   


#================== linear regression ========================================


hashtag_set=['#GoHawks','#GoPatriots','#NFL','#Patriots','#SB49','#SuperBowl']

for hashtag in hashtag_set:
    data = pd.read_csv('./Extracted_data/Q2_'+hashtag+'.csv')
    
    #===================== One-hot encoding =====================
    time_of_day_set = range(0,24)
    for time_of_day in time_of_day_set:
        time_of_day_column_to_be_added = []
        for time_of_day_item in data['time_of_the_day']:
            if time_of_day_item == time_of_day:
                time_of_day_column_to_be_added.append(1)
            else:
                time_of_day_column_to_be_added.append(0)
        data.insert(data.shape[1]-1,
                    str(time_of_day)+'th_hour',
                    time_of_day_column_to_be_added)


    feature = data.copy()
    feature.drop('time_windows', 1, inplace = True)
    feature.drop('time_of_the_day', 1, inplace = True)
    target = feature.pop('target_value')
    

    lin_reg = LinearRegression(fit_intercept = False)
    lin_reg_result = lin_reg.fit(feature, target)
    
    predicted_values = lin_reg_result.predict(feature)
    
    RMSE = np.sqrt(mean_squared_error(predicted_values, target))
    
    print '======================================='
    print 'Processing hashtag "' + hashtag + '"......'
    
    print 'RMSE = '+ str(RMSE)
    
    #==================== Perform t-test and calculate p value ====================


    LR_model = sm.OLS(target, feature)
    LR_results = LR_model.fit()
    print  '=========== t-test results: ============'
    print LR_results.summary()
    print '=======================================\n'
    print 'P values: ' 
    print LR_results.pvalues
    print 'T values: ' 
    print LR_results.tvalues
    

    

