import json, os
import matplotlib.pyplot as plt
import datetime, time
import pytz
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

test_data_path = "/Volumes/DATA/Documents/Term 10/219 Project5/test_data"
pst_tz = pytz.timezone('US/Pacific')

featuresTable5h = pd.read_pickle('/Volumes/DATA/Documents/Term 10/219 Project5/allHashtagsFeatures.pickle')
algo = linear_model.LinearRegression()
# algo = RandomForestRegressor()
train_FeatureSet = featuresTable5h[['num_of_tweets_0h', 'num_of_tweets_1h', 'num_of_tweets_2h', 'num_of_tweets_3h','num_of_tweets_4h',
                                    'total_num_of_retweets_0h','total_num_of_retweets_1h','total_num_of_retweets_2h','total_num_of_retweets_3h','total_num_of_retweets_4h',
                                    'sum_of_followers_0h','sum_of_followers_1h','sum_of_followers_2h','sum_of_followers_3h','sum_of_followers_4h',
                                    'max_num_of_followers_0h','max_num_of_followers_1h','max_num_of_followers_2h','max_num_of_followers_3h','max_num_of_followers_4h',
                                    'time_of_the_day_0h','time_of_the_day_1h','time_of_the_day_2h','time_of_the_day_3h','time_of_the_day_4h'
                                    ]]
train_ResultSet = featuresTable5h['target_value']
algo.fit(train_FeatureSet, train_ResultSet)

# ----------------------------------Build Feature Table of Test Files-------------------------------------


for file_name in os.listdir(test_data_path):
    if file_name.endswith(".txt"):
        time_stamps = []
        num_of_followers = []
        num_of_retweets = []
        time_of_the_day = []
        time_windows = []
        with open(os.path.join(test_data_path,file_name)) as text:
            print "loaded " + file_name
            # Pad sample8_period1.text to 6 hours
            if file_name == 'sample8_period1.txt':
                time_stamps = [1422486005]
                num_of_followers = [0]
                num_of_retweets = [0]
                time_of_the_day = [15]

            for line in text:
                data = json.loads(line)
                time_stamps.append(data['firstpost_date'])
                num_of_retweets.append(data['metrics']['citations']['total'])
                num_of_followers.append(int(data['author']['followers']))
                time_of_the_day.append(datetime.datetime.fromtimestamp(data['firstpost_date'], pst_tz).hour)

        first_hour_timestamp = int(time.mktime(
            datetime.datetime.fromtimestamp(min(time_stamps), pst_tz).replace(microsecond=0, second=0,
                                                                              minute=0).timetuple()))

        for time_stamp in time_stamps:
            time_windows.append((time_stamp - first_hour_timestamp) / 3600)

        test_df = pd.DataFrame({'time_stamp': time_stamps,
                                'num_of_retweets': num_of_retweets,
                                'num_of_followers': num_of_followers,
                                'time_of_the_day': time_of_the_day,
                                'time_window': time_windows})
        num_of_tweets = []
        total_num_of_retweets = []
        sum_of_followers = []
        max_num_of_followers = []
        time_of_the_day = []

        for time_window in range(max(time_windows)):
            data_this_time_window = test_df.loc[test_df['time_window'] == time_window]
            num_of_tweets.append(data_this_time_window.shape[0])
            total_num_of_retweets.append(data_this_time_window['num_of_retweets'].sum())
            sum_of_followers.append(data_this_time_window['num_of_followers'].sum())
            max_num_of_followers.append(0 if data_this_time_window['num_of_followers'].empty else max(
                data_this_time_window['num_of_followers']))
            time_of_the_day.append(time_window - 24 * (time_window / 24) +
                                   test_df.loc[test_df['time_window'] == 0]['time_of_the_day'].values[0])

        testFeaturesTable5h = pd.DataFrame(
            {'num_of_tweets_0h': num_of_tweets[4:], 'num_of_tweets_1h': num_of_tweets[3:-1],
             'num_of_tweets_2h': num_of_tweets[2:-2],
             'num_of_tweets_3h': num_of_tweets[1:-3], 'num_of_tweets_4h': num_of_tweets[0:-4],
             'total_num_of_retweets_0h': total_num_of_retweets[4:],
             'total_num_of_retweets_1h': total_num_of_retweets[3:-1],
             'total_num_of_retweets_2h': total_num_of_retweets[2:-2],
             'total_num_of_retweets_3h': total_num_of_retweets[1:-3],
             'total_num_of_retweets_4h': total_num_of_retweets[0:-4],
             'sum_of_followers_0h': sum_of_followers[4:], 'sum_of_followers_1h': sum_of_followers[3:-1],
             'sum_of_followers_2h': sum_of_followers[2:-2],
             'sum_of_followers_3h': sum_of_followers[1:-3], 'sum_of_followers_4h': sum_of_followers[0:-4],
             'max_num_of_followers_0h': max_num_of_followers[4:], 'max_num_of_followers_1h': max_num_of_followers[3:-1],
             'max_num_of_followers_2h': max_num_of_followers[2:-2],
             'max_num_of_followers_3h': max_num_of_followers[1:-3],
             'max_num_of_followers_4h': max_num_of_followers[0:-4],
             'time_of_the_day_0h': time_of_the_day[4:], 'time_of_the_day_1h': time_of_the_day[3:-1],
             'time_of_the_day_2h': time_of_the_day[2:-2],
             'time_of_the_day_3h': time_of_the_day[1:-3], 'time_of_the_day_4h': time_of_the_day[0:-4],
             'target_value': test_df.loc[test_df['time_window'] == 5].shape[0],
             })

        test_FeatureSet = testFeaturesTable5h[
            ['num_of_tweets_0h', 'num_of_tweets_1h', 'num_of_tweets_2h', 'num_of_tweets_3h', 'num_of_tweets_4h',
             'total_num_of_retweets_0h', 'total_num_of_retweets_1h', 'total_num_of_retweets_2h',
             'total_num_of_retweets_3h', 'total_num_of_retweets_4h',
             'sum_of_followers_0h', 'sum_of_followers_1h', 'sum_of_followers_2h', 'sum_of_followers_3h',
             'sum_of_followers_4h',
             'max_num_of_followers_0h', 'max_num_of_followers_1h', 'max_num_of_followers_2h', 'max_num_of_followers_3h',
             'max_num_of_followers_4h',
             'time_of_the_day_0h', 'time_of_the_day_1h', 'time_of_the_day_2h', 'time_of_the_day_3h',
             'time_of_the_day_4h'
             ]]

        print file_name, 'target:', test_df.loc[test_df['time_window'] == 5].shape[0], 'predicted:', int(
            algo.predict(test_FeatureSet)[0])