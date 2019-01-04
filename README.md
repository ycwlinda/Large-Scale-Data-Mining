# Large-Scale-Data-Mining  EE219 Projects

**Project 1 Classification Analysis on Textual Data**
---
In this project, we work with “20 Newsgroups” dataset. It is a collection of approximately 20,000 newsgroup documents, partitioned evenly across 20 different newsgroups, each corresponding to a different topic. 

Each newsgroup document is a textual string that contains raw information. 

Our goal is to correctly classify all the documents into their corresponding classification. 

The documents were mostly classified using several classifiers such as Naive Bayesian, SVMs and Logistic Regressors.

**Project 2 Clustering**
---
The goal includes:
- To find proper representations of the data, so that the clustering is efficient and gives out reasonable results.
- To perform K-means clustering on the dataset, and evaluate the performance of the clustering.
- To try different preprocess methods which may increase the performance of the clustering.

We applied the clustering method on the dataset, and we tried the K-means clustering method with different dimension reduction, optimized the dimension reduction separately and compared the performance with each other. 

We started with 2-cluster case and then moved on to the multiple cluster case and gave a detailed analysis on each of the outcome results.

**Project 3 Collaborative Filtering**
---
In this project, we use different methods to build recommendation system on the ratings of movies.

The basic idea under the recommendation system is that we use user-item relation, which refers to collaborative filtering, to infer their interests.

We applied Neighborhood-based, Model-based and Naive three different collaborative filtering methods to build recommendation system and compare their performance.

**Project 4 Regression Analysis**
---
In this project, we have done data analysis on Network Backup dataset which is analyzed by using the data mining approaches of regression.
The "network_backup_dataset" has captured simulated traffic data on a backup system and contains information of the size of the data moved to the destination as well as the time it took for backup. 

Our mission was to predict the backup size of the traffic depending on the file-name, day/time of backup. 

Prediction models have been using Linear, Random Forest, Neural Network Polynomial and knn Regression.

We used all attributes, except Backup time, as candidate features for the prediction of backup size.

**Project 5 Popularity Prediction on Twitter**
---
In this project, twitter data is collected by querying popular hashtags related to the Super Bowl spanning a period starting from 2 weeks before the game to a week after the game. 

We used these data to train a regression model and then use the model to making predictions for other hashtags. 

The test data consists of tweets containing a hashtag in a specified time window, and we have then used our model to predict the number of tweets containing the hash-tag posted within one hour immediately following the given time window. 

Finally, we used the knowing data to define our problem like sentiment analysis and try to implement our idea and show how to work.



