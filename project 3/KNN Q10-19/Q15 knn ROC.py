import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset,Reader,accuracy,KNNWithMeans
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('/Volumes/DATA/Downloads/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))

thre_set = [2.5, 3, 3.5, 4]
subplotindex = 220
plt.figure(figsize=[12, 10]).set_tight_layout(True)

for thre in thre_set:
    trainset, testset = train_test_split(data, test_size=0.1)
    algo = KNNWithMeans(k=30,sim_options={'name': 'pearson'})  # find in Q11
    predictions = algo.fit(trainset).test(testset)
    predicted_ratings = [t[3] for t in predictions]
    ground_truth_ratings = [t[2] for t in testset]
    ground_truth_labels = [1 if t >= thre else 0 for t in ground_truth_ratings]
    # Plot ROC
    fpr, tpr, _ = roc_curve(ground_truth_labels, predicted_ratings)
    roc_auc = auc(fpr, tpr)
    lw = 2
    subplotindex = subplotindex + 1
    plt.subplot(subplotindex)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, Threshold = {}'.format(thre))
    plt.legend(loc="lower right")
