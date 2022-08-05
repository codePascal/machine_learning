"""
Introduction to Machine Learning
Task 1b
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Mar, 2021

The task is transform the given input vectors with given feature transformations
and to predict the labels using linear regression.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

# Regularization parameters
rid_lambdas = np.linspace(0.01,100,1000)
elnet_lambdas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
las_lambdas = np.linspace(0.001,0.01,100)

# Load data
full_data = pd.read_csv("handout/train.csv")
y_data = full_data.values[:, 1]
x_data = full_data.values[:, 2:]

# Transform features
x_transformed = np.hstack((x_data,
                           np.square(x_data),
                           np.exp(x_data),
                           np.cos(x_data),
                           np.ones((x_data.shape[0],1))))

# Fit linear model with Ridge regression and cross validation
rid_model = RidgeCV(alphas=rid_lambdas,fit_intercept=False,cv=5)
rid_model.fit(x_transformed, y_data)

# lin_model = LinearRegression(fit_intercept=False)
# lin_model.fit(x_transformed, y_data)
#
# elnet_model = ElasticNetCV(l1_ratio=elnet_lambdas,n_alphas=100,fit_intercept=False)
# elnet_model.fit(x_transformed,y_data)
#
# las_model = LassoCV(alphas=las_lambdas,cv=2,fit_intercept=False) # Default: 5 folds
# las_model.fit(x_transformed, y_data)
# las_mean_error = np.mean(las_model.mse_path_,1)

# Store results
np.savetxt("rigde_submission.csv",rid_model.coef_,comments='',delimiter=",",fmt="%s")
# np.savetxt("linear.csv",lin_model.coef_,comments='',delimiter=",",fmt="%s")
# np.savetxt("elnet.csv",elnet_model.coef_,comments='',delimiter=",",fmt="%s")
# np.savetxt("lasso.csv",las_model.coef_,comments='',delimiter=",",fmt="%s")
