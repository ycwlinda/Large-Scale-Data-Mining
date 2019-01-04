import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset,Reader,accuracy,KNNWithMeans,NMF,SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('/Volumes/DATA/Downloads/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))

def plot_ROC_of_algo(algo, curvelabel, color, trainset, testset):
    predictions = algo.fit(trainset).test(testset)
    predicted_ratings = [t[3] for t in predictions]
    ground_truth_ratings = [t[2] for t in testset]
    ground_truth_labels = [1 if t >= 3 else 0 for t in ground_truth_ratings]
    fpr, tpr, _ = roc_curve(ground_truth_labels, predicted_ratings)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color,
             lw=2, label=(curvelabel,'(area = %0.4f)' % roc_auc))


plt.figure(figsize=[12, 10]).set_tight_layout(True)
trainset, testset = train_test_split(data, test_size=0.1)
algo = KNNWithMeans(k=30,sim_options={'name': 'pearson'})  # find in Q11
plot_ROC_of_algo(algo=algo, curvelabel='K-NN', color='darkorange', trainset=trainset, testset=testset)
algo = NMF(n_factors=20)
plot_ROC_of_algo(algo=algo, curvelabel='NNMF', color='cyan', trainset=trainset, testset=testset)
algo = SVD(n_factors=8, biased=True)
plot_ROC_of_algo(algo=algo, curvelabel='MF with bias', color='lime', trainset=trainset, testset=testset)


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
