import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, AlgoBase, dataset
from surprise.model_selection import cross_validate

class NaiveFilter(AlgoBase):
    def __init__(self,dataset):
        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.dataset = dataset.build_full_trainset()
    def estimate(self, u, i):
        raw_uid = self.trainset.to_raw_uid(u)
        return np.mean([r for (_, r) in self.dataset.ur[self.dataset.to_inner_uid(raw_uid)]])


# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('/Volumes/DATA/Downloads/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))

algo = NaiveFilter(dataset = data)
# cross-validation and print results.
cv_result = cross_validate(algo, data, measures=['RMSE'], cv=10, verbose=True, n_jobs=-1)
