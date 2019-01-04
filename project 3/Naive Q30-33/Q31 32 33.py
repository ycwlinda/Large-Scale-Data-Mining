import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset,Reader,accuracy,AlgoBase
from surprise.model_selection import cross_validate,KFold
import pandas as pd

# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('/Volumes/DATA/Downloads/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))

class NaiveFilter(AlgoBase):
    def __init__(self,dataset):
        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.dataset = dataset.build_full_trainset()
    def estimate(self, u, i):
        raw_uid = self.trainset.to_raw_uid(u)
        return np.mean([r for (_, r) in self.dataset.ur[self.dataset.to_inner_uid(raw_uid)]])

# define a cross-validation iterator
kf = KFold(n_splits=10)

algo = NaiveFilter(dataset = data)
this_k_RMSE = list()
# use cross-validation iterator, perform CV manually
for trainset, testset in kf.split(data):
    # fit the whole trainset
    algo.fit(trainset)

    #trim testset here
    testset_df = pd.DataFrame(testset, columns=['userId','movieId','rating'])
    testset_popular = testset_df.groupby("movieId").filter(lambda x: len(x) > 2).values.tolist()
    # testset_unpopular = testset_df.groupby("movieId").filter(lambda x: len(x) <= 2).values.tolist()
    # testset_highvariance =testset_df.groupby("movieId").filter(lambda x: np.var(x['rating'])>=2 and len(x)>=5 ).values.tolist()

    predictions = algo.test(testset_popular)
    this_k_RMSE.append(accuracy.rmse(predictions, verbose=True))

print 'average RMSE =',(np.mean(this_k_RMSE))


