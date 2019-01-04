
# coding: utf-8

# In[5]:

import os
import numpy as np
from sklearn.metrics import precision_score,recall_score
import matplotlib.pyplot as plt
from surprise import Dataset,Reader,KNNWithMeans,Trainset,NMF,SVD
#from suprise.prediction_algorithms.matrix_factorization import NMF,SVD
from surprise.model_selection import cross_validate,train_test_split,KFold
from collections import defaultdict,Counter
from itertools import izip
import operator

# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('E:/ee219/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))
print('load finish')
#knnoptm=30
#nnmfoptm=20
#mfoptm=8
trainset, testset = train_test_split(data, test_size=0.1)

#def make_prediction(filter_type):
    #if filter_type=='KNN':
sim_options={'name': 'pearson','user_based':True}
algo=SVD(n_factors=8,biased=True)
    #elif filter_type='NNMF':
        #filter_=NMF(n_factors=20)
    #elif filter_type='MFbias':
#filter_=SVD(n_factors=8,biased=True)
    #filter_.fit(trainset) 
algo.fit(trainset)
predictions=algo.test(testset)
    
truth_table=np.array([prediction.r_ui for prediction in predictions])
truth_table_copy=truth_table.copy()
truth_table_copy[truth_table<3]=0
truth_table_copy[truth_table>=3]=1

user_dic=defaultdict(list)
for uid,_,tru_r,est,_ in predictions:
    user_dic[uid].append((est,tru_r))
print 'dic created'
def precision_recall(predictions,t):
    precision=dict()
    recall=dict()

    for uid, user_ratings in user_dic.items():
        if len(user_ratings)<t:
            continue
        user_ratings.sort(key=lambda x:x[0],reverse=True)

        G=sum((tru_r>=3) for (_,tru_r) in user_ratings)
        #print G
        if G==0:
            continue

        St=t

        #intersection
        g_s=sum((tru_r>=3) for (_,tru_r) in user_ratings[:t])

        precision[uid]=float(g_s)/float(St)
        recall[uid]=float(g_s)/float(G)
    return precision,recall
precision_t=[]
recall_t=[]
t_value=range(1,26)
for t in t_value:
    print ("t_value= %d" %t)
    kf=KFold(n_splits=10)
    kf_precision=[]
    kf_recall=[]
    precision,recall=precision_recall(predictions,t)
    kf_precision.append(float(sum(precision.values()))/len(precision))
    kf_recall.append(float(sum(recall.values()))/len(recall))
    precision_t.append(np.mean(kf_precision))
    recall_t.append(np.mean(kf_recall))
#print precision_t
#print recall_t
plt.plot(t_value,precision_t,lw=2.5)
plt.xlabel("t_value")
plt.ylabel("Average precision score")
plt.title("SVD, Average Precision against t")
plt.show()

plt.plot(t_value,recall_t,lw=2.5)
plt.xlabel("t_value")
plt.ylabel("Average Recall score")
plt.title("SVD, Average Recall against t")
plt.show()


# In[ ]:



