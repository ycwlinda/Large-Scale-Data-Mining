
import os
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

#===================== Function extracting data from file ===================== 
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
    
    number_of_url_citations = []
    author_names = []
    number_of_mentions = []
    ranking_scores = []
    number_of_hashtags = []
    
    # ----------------------- Extract data from file ---------------------------
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
        
        number_of_url_citations.append(len(data['tweet']['entities']['urls']))
        author_names.append(data['author']['nick'])
        number_of_mentions.append(len(data['tweet']['entities']['user_mentions']))
        ranking_scores.append(data['metrics']['ranking_score'])
        number_of_hashtags.append(data['title'].count('#'))
            
    input_file.close()

    #-------------------- Calculate parameters based on hour index --------------------
    total_hours = int((max(time_stamps)-min(time_stamps))/3600)+1
    hourly_number_of_tweets = [0] * total_hours
    hourly_number_of_retweets = [0] * total_hours
    hourly_sum_of_followers = [0] * total_hours
    hourly_max_number_of_followers = [0] * total_hours
    hourly_time_of_the_day = [0] * total_hours
    time_windows = [0] * total_hours
    start_time = min(time_stamps)
    
    
    
    hourly_number_of_url_citations = [0] * total_hours
    hourly_number_of_authors = [0] * total_hours
    hourly_author_set = [0] * total_hours
    for i in range(0, total_hours):
        hourly_author_set[i] = set([])
    hourly_number_of_mentions = [0] * total_hours
    hourly_total_ranking_scores = [0] * total_hours
    hourly_number_of_hashtags = [0] * total_hours
    
    
    
    for i in range(0, len(time_stamps)):
        current_hour = int((time_stamps[i]-start_time)/3600)
        
        hourly_number_of_tweets[current_hour] += 1
        if is_retweet[i]:
            hourly_number_of_retweets[current_hour] += 1
                                      
        hourly_sum_of_followers[current_hour] += followers_of_users[i]
    
        if followers_of_users[i] > hourly_max_number_of_followers[current_hour]:
            hourly_max_number_of_followers[current_hour] = followers_of_users[i]
        
        
        hourly_number_of_url_citations[current_hour] += number_of_url_citations[i]
        hourly_author_set[current_hour].add(author_names[i])
        hourly_number_of_mentions[current_hour] += number_of_mentions[i]
        hourly_total_ranking_scores[current_hour] += ranking_scores[i]
        hourly_number_of_hashtags[current_hour] += number_of_hashtags[i]


    for i in range(0, len(hourly_author_set)):
        hourly_number_of_authors[i] = len(hourly_author_set[i])
    
    for i in range(0, total_hours):
        time_windows[i] = i
        hourly_time_of_the_day[i] = i%24

    
    #------------------ Build DataFrame and save to file ----------------------
    target = hourly_number_of_tweets[1:]
    target.append(0)
    data = np.array([
                time_windows,
                hourly_time_of_the_day,
                hourly_number_of_tweets,
                hourly_number_of_retweets,
                hourly_sum_of_followers,
                hourly_max_number_of_followers,
                
                hourly_number_of_url_citations,
                hourly_number_of_authors,
                hourly_number_of_mentions,
                hourly_total_ranking_scores,
                hourly_number_of_hashtags,
                target])
    
    data = np.transpose(data)
    df = DataFrame(data)
    df.columns = [
                  'time_windows',
                  'time_of_the_day',
                  'num_tweets', 
                  'num_retweets', 
                  'sum_followers',
                  'max_followers',
                
                  'num_URLs',
                  'num_authors',
                  'num_mensions',
                  'ranking_score',
                  'num_hashtags',
                  'target']

    if os.path.isdir('./Extracted_data'):
        pass
    else:
        os.mkdir('./Extracted_data')
    df.to_csv('./Extracted_data/Q3_'+hashtag+'.csv', index = False)   
#==============================================================================


       #================== linear regression===================

from sklearn.metrics import mean_squared_error

hashtag_set=['#GoHawks','#GoPatriots','#NFL','#Patriots','#SB49','#SuperBowl']


for hashtag in hashtag_set:
    data = pd.read_csv('./Extracted_data/Q3_'+hashtag+'.csv')
  
    #---------------------------- One-hot encoding ----------------------------
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
      
    #-------------------------------------------------------
    feature= data.copy()
    
    feature.drop('time_of_the_day', 1, inplace = True)
    feature.drop('time_windows', 1, inplace = True)
    target = feature.pop('target')
    
    lin_reg = LinearRegression(fit_intercept = False)
    lin_reg_result = lin_reg.fit(feature, target)
    
    predicted_values = lin_reg_result.predict(feature)
    
    print '#######################################################'
    print 'Processing hashtag "' + hashtag + '"......'
    
    RMSE = np.sqrt(mean_squared_error(predicted_values, target))
    print 'RMSE = '+ str(RMSE)
    #----------------------------- Perform t-test -----------------------------
   
    LR_model = sm.OLS(target, feature)
    LR_results = LR_model.fit()
    print (LR_results.summary())
    print 'P values: ' 
    print LR_results.pvalues
    print 'T values: '
    print LR_results.tvalues
    
    p = LR_results.pvalues
    pdf = DataFrame(p)
    pdf.columns = ['p-value']
    pdf_9_feature= pdf.iloc[0:9]
    pdf_sorted = pdf_9_feature.sort_values(by=['p-value'])
    important_set= pdf_sorted.iloc[0:3]
    three_feature_set = important_set.index.tolist()

    print 'The most 3 important features for '+hashtag+'are ...'
    print important_set
    
    for f in three_feature_set:
        plt.clf
        fig, ax = plt.subplots()
        ax.scatter(feature[f], predicted_values)
        #ax.plot([0,max(target_data)], [0,0], 'k--', lw = 4)
        ax.set_xlabel(str(f), fontsize = 10)
        ax.set_ylabel('Predicted value', fontsize = 10)
        plt.title('Predicted value vs. '+str(f)+ '(' + hashtag + ')', fontsize = 15)
        #plt.axis([0,10000,-1200,400])
        plt.show()
    
  