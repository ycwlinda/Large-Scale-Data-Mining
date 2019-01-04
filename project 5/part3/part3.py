import json
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime, time
import pytz
import pandas as pd
import string
from sklearn import svm, linear_model
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# 0 - WA, 1 - MA, 2 - others
def checkLocation(locationStr):
    WashingtonLocations = ['Seattle', 'Washington', 'WA', 'Kirkland']
    MassachusettsLocations = ['Massachusetts', 'Boston', 'MA', 'Springfield', 'Mass.']
    WA_black_list = ["dc", "d.c.", "d.c."]
    for subStr in WashingtonLocations:
        if subStr in locationStr:
            for subStr2 in WA_black_list:
                if subStr2 not in locationStr:
                    return 0
    for subStr in MassachusettsLocations:
        if subStr in locationStr:
            return 1
    return 2

pst_tz = pytz.timezone('US/Pacific')
df = pd.DataFrame()

titles = []
locations = []
locationStrs = []
time_stamps = []
time_windows = []

tweet_data_path = "/Volumes/DATA/Documents/Term 10/219 Project5/tweet_data"
for file_name in os.listdir(tweet_data_path):
    if file_name.endswith(".txt"):
        with open(os.path.join(tweet_data_path,file_name)) as text:
            print "loaded " + file_name
            for line in text:
                data = json.loads(line)
                locationCode = checkLocation(data['tweet']['user']['location'])
                if (locationCode != 2):
                    titles.append("".join(b for b in data['title'] if ord(b) < 128))
                    locations.append(locationCode)
                    time_stamps.append(data['citation_date'])
                    locationStrs.append(data['tweet']['user']['location'])

first_hour_timestamp = int(time.mktime(datetime.datetime.fromtimestamp(min(time_stamps), pst_tz).replace(microsecond=0,second=0,minute=0).timetuple()))
for time_stamp in time_stamps:
    time_windows.append((time_stamp - first_hour_timestamp)/3600)

df = pd.DataFrame({'titles':titles, 'locations':locations, 'time_window':time_windows})

# df = pd.read_pickle('/Volumes/DATA/Documents/Term 10/219 Project5/locationTweets.pickle')
sid = SentimentIntensityAnalyzer()

WASentimentList = []
MASentimentList = []

for time_window in range(max(df['time_window'])):
    WASentimentListThisHour = []
    MASentimentListThisHour = []
    tweetThisHour = df.loc[df['time_window']==time_window]
    for row,tweet in tweetThisHour.iterrows():
        if tweet['locations'] == 0:
            WASentimentListThisHour.append(sid.polarity_scores(tweet['titles'])['compound'])
        if tweet['locations'] == 1:
            MASentimentListThisHour.append(sid.polarity_scores(tweet['titles'])['compound'])
    WASentimentList.append(np.mean(WASentimentListThisHour))
    MASentimentList.append(np.mean(MASentimentListThisHour))

plt.plot(range(max(df['time_window'])),WASentimentList,label='WA Sentiment')
plt.plot(range(max(df['time_window'])),MASentimentList,label='MA Sentiment')
plt.legend(loc="upper right")