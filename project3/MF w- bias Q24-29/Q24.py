import os
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset,Reader,SVD
from surprise.model_selection import cross_validate

# Load the dataset. The path needs to be set to where ml-latest-small is downloaded
file_path = os.path.expanduser('/Volumes/DATA/Downloads/ml-latest-small/ratings.csv')
data = Dataset.load_from_file(file_path, reader=Reader(line_format='user item rating timestamp', sep=',', skip_lines=1))

k_values = np.arange(2,51,2)
RMSE = list()
MAE = list()

for k in k_values:
    algo = SVD(n_factors=k, biased=True)
    # cross-validation and print results.
    cv_result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True, n_jobs=-1)
    RMSE.append(np.average(cv_result.get('test_rmse')))
    MAE.append(np.average(cv_result.get('test_mae')))

plt.figure(figsize=[6,5]).set_tight_layout(True)
plt.plot(k_values, RMSE, label='RMSE')
plt.plot(k_values, MAE, label='MAE')
plt.xlabel('k')
plt.scatter((MAE.index(min(MAE))+1)*2, min(MAE))
print ((MAE.index(min(MAE))+1)*2, min(MAE))
plt.scatter((RMSE.index(min(RMSE))+1)*2, min(RMSE))
print ((RMSE.index(min(RMSE))+1)*2, min(RMSE))
plt.legend(loc="upper right")
