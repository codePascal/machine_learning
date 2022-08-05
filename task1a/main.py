"""
Introduction to Machine Learning
Task 1b
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Mar, 2021

The task is to perform 10-fold cross-validation with ridge regression for given
lambdas and to report the root mean squared error (RMSE) averaged over the 10
test folds.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# parameters
K = 10 # number of folds
lambdas = [0.1, 1, 10, 100, 200] # ridge parameter values
error_vec = [] # vec to store RMSE values

# load data
data_full = pd.read_csv('handout/train.csv')

# extract features and labels
X_full = np.matrix(data_full[['x1','x2','x3','x4','x5','x6','x7','x8','x9',
                              'x10','x11','x12','x13']].values)
Y_full = np.transpose(np.matrix(data_full['y'].values))

# iterate over all lambdas
for lam in lambdas:
    mean_error = 0

    # perform CV
    kf = KFold(n_splits=K, shuffle=True, random_state=123)
    model = Ridge(alpha=lam, tol=1e-6)
    for (train,test) in kf.split(X_full,y=Y_full):
        # fit model
        model.fit(X=X_full[train],y=Y_full[train])

        # predict labels
        prediction = model.predict(X_full[test])

        # compute rmse
        error = mean_squared_error(Y_full[test],prediction)**0.5
        mean_error = mean_error + 1/K*error

    error_vec.append(mean_error)

print(error_vec)
np.savetxt('RMSE.csv', error_vec, delimiter=',')
