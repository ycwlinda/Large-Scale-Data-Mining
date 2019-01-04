import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset,Reader,NMF
import csv

# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('/Volumes/DATA/Downloads/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))
full_trainset = data.build_full_trainset()

with open('/Volumes/DATA/Downloads/ml-latest-small/movies.csv') as f:
    movieGenres=[tuple(line) for line in csv.reader(f)]

algo = NMF(n_factors=20)
algo.fit(full_trainset)

col2exam = 5
Vcol = algo.qi[:,col2exam]
sortedCol = Vcol.argsort()
topTen = [full_trainset.to_raw_iid(r) for r in sortedCol[sortedCol.shape[0]-10:sortedCol.shape[0]]]

for movie in topTen:
    print [item[2] for item in movieGenres if item[0] == movie]