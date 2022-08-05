"""
Introduction to Machine Learning
Dummy task 0: Linear regression
Team Naiveoutliers
March 2021
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# load data with pandas
train_data = pd.read_csv("handout/train.csv")
test_data = pd.read_csv("handout/test.csv")

# extract training data
y_train = train_data.values[:, 1]
x_train = train_data.values[:, 2:]

# create linear regression model
model = LinearRegression()

# fit the model to the training data
model.fit(x_train, y_train)

# extract test data
id_test = test_data.values[:, 0]
x_test = test_data.values[:, 1:]

# compute test y (mean of x)
y_test = np.mean(x_test, 1)

# predict output
y_pred = model.predict(x_test)

# compute and print RMSE
RMSE = mean_squared_error(y_test, y_pred)**0.5
print(RMSE)

# write to output file
output = np.asarray([id_test, y_pred]).transpose()
np.savetxt("submission.csv", output, header="Id,y", delimiter=",", fmt="%i,%s")
