"""
Introduction to Machine Learning
Task 2
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Apr, 2021

TASK INTRODUCTION

Applying ML approaches to real-world problems requires some extra steps in
handling specific data artifacts. Some common challenges you may face are
missing values, an imbalance of the labels, or heavy-tailed distributions in the
data. These can be addressed through a specific choice of pre-processing and/or
model components.

In this task, as an illustration of a real-world problem, you are asked to
predict the evolution of hospital patients' states and needs during their stay
in the Intensive Care Unit (ICU). For each patient, you are provided with the
monitoring information collected during the first 12h of their stay. From the
data, you first need to extract meaningful features describing this 12h stay.
Then, for each sub-task, you should select an appropriate model to train on your
pre-processed data. You will face the typical challenges of working with real
medical data: missing features and imbalanced classification, predicting
rarely-occuring events

DATA DESCRIPTION

The training (train_features.csv) and test set (test_features.csv) are both
arranged in the same way. Each row contains the information for a patient at a
given time identified by a unique patient ID and timestamp in columns pid and
Time. The rest of the columns are medical information, either vital signs such
as the Heart rate, or lab tests such as Calcium level in the patient blood.
Finally, you are provided with the Age of the patient which is the same during
the entire stay.

All medical measurements are not available at each timestep, meaning the data
contains a lot of missing values, indicated with ‘nan’ in our case. To simplify
the problem, the data is already re-sampled hourly. This means that we aggregate
measurements by one-hour period, thus there are only 12 rows for a given patient
in the corresponding .csv file.

The last .csv file, namely train_labels.csv, contains the ground truth labels of
the training set for each subtask. Each row is identified by a unique pid
corresponding to a patient from the training set.

SUBTASK DESCRIPTIONS

We now detail the three types of sub-tasks you are asked to solve. In all
sub-tasks, we want to predict an aspect of the evolution of a patient’s state
from the first 12 hours of his stay in the ICU.

SUB-TASK 1: ORDERING OF MEDICAL TEST
Here we are interested in anticipating the future needs of the patient. You have
to predict whether a certain medical test is ordered by a clinician in the
remaining stay. This sub-task is a binary classification : 0 means that there
will be no further tests of this kind ordered whereas 1 means that at least one
is ordered in the remaining stay.

The corresponding columns containing the binary ground truth in
train_labels.csv are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST,
LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI,
LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.

Because there is an imbalance between labels in these sub-tasks we evaluate the
performance of a model with the Area Under the Receiver Operating Characteristic
Curve, which is a threshold-based metric. To achieve good performance, it is
important to produce (probabilistic) real-valued predictions in the interval
[0, 1].

SUB-TASK 2: SEPSIS PREDICTION
In this sub-task, we are interested in anticipating future life-threatening
events. You have to predict whether a patient is likely to have a sepsis event
in the remaining stay. This task is also a binary classification : 0 means that
no sepsis will occur, 1 otherwise.

The corresponding column containing the binary ground-truth in
train_labels.csv is LABEL_Sepsis.

This task is also imbalanced, thus we’ll also evaluate performance using Area
Under the Receiver Operating Characteristic Curve.

SUB-TASK 3: KEYS VITALS SIGNS PREDICTION
In this type of sub-task, we are interested in predicting a more general
evolution of the patient state. To this effect, here we aim at predicting the
mean value of a vital sign in the remaining stay. This is a regression task.

The corresponding columns containing the real-valued ground truth in
train_labels.csv are: LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate.

To evaluate the performance of a given model on this sub-task we use R2-Score.
"""

import pandas as pd
import numpy as np
import scipy
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import ElasticNet
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc

############
# USER INPUT
############
use_preprocessed = False

# get headers for Task 1
classification_labels = [
"LABEL_BaseExcess",
"LABEL_Fibrinogen",
"LABEL_AST",
"LABEL_Alkalinephos",
"LABEL_Bilirubin_total",
"LABEL_Lactate",
"LABEL_TroponinI",
"LABEL_SaO2",
"LABEL_Bilirubin_direct",
"LABEL_EtCO2"
]
classification_features = [c[6:] for c in classification_labels]

# get headers for Task 2
sepsis_label = "LABEL_Sepsis"

# get headers for Task 3
regression_labels = [
"LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"
]

# create vector to save the scores
ROC_scores = []

if not use_preprocessed:
    df_train_features = pd.read_csv("/handout/train_features.csv")
    df_train_labels = pd.read_csv("/handout/train_labels.csv")
    df_test_features = pd.read_csv("/handout/test_features.csv")

    # ================================
    # load data into pandas dataframes
    # ================================
    print('loading raw data...')
    df_train_features = df_train_features.set_index(["pid"]).sort_values(['pid','Time'])
    df_test_features = df_test_features.set_index(["pid"]).sort_values(['pid','Time'])
    df_train_labels = df_train_labels.set_index(["pid"]).sort_values(['pid'])
    print('...done')

    feature_scaler = StandardScaler()
    NNImputer = KNNImputer(n_neighbors=10)
    MeanImputer = SimpleImputer(strategy='mean')

    # ==========================
    # Preprocessing training set
    # ==========================
    print('processing training set...')
    # eliminate time dependency:
    df_train_features.drop(labels='Time',axis=1,inplace=True)

    # we average all values and append the gradient of the following columns:
    # RRate, Heartrate, ABPm, ABPd, SpO2
    for column in ['RRate','Heartrate','ABPm','ABPd','SpO2']:
        # fill missing values
        df_train_features[column].groupby('pid',sort=False).fillna(method='ffill',inplace=True)
        df_train_features[column].groupby('pid',sort=False).fillna(method='bfill',inplace=True)
        # get the differences
        df_train_features[str('gradient_'+column)] = df_train_features[column].groupby('pid',sort=False).diff()

    # the average of the differences is the gradient
    df_train_features = df_train_features.groupby('pid',sort=False).mean()

    # replace missing values by KNN-imputation
    X_train = MeanImputer.fit_transform(df_train_features.values)

    # apply standard scaler
    X_train = feature_scaler.fit_transform(X_train)
    df_train_features[df_train_features.columns] = X_train

    print('...done')

    # =========================
    # Preprocessing of test set
    # =========================
    print('processing test set...')
    # eliminate time dependency
    df_test_features.drop(labels='Time',axis=1,inplace=True)

    # we average all values and append the gradient of the following columns:
    # RRate, Heartrate, ABPm, ABPd, SpO2
    for column in ['RRate','Heartrate','ABPm','ABPd','SpO2']:
        # fill missing values
        df_test_features[column].groupby('pid',sort=False).fillna(method='ffill',inplace=True)
        df_test_features[column].groupby('pid',sort=False).fillna(method='bfill',inplace=True)
        # get the differences
        df_test_features[str('gradient_'+column)] = df_test_features[column].groupby('pid',sort=False).diff()

    # the average of the differences is the gradient
    df_test_features = df_test_features.groupby('pid',sort=False).mean()

    # replace missing values by KNN-imputation
    X_test = MeanImputer.transform(df_test_features.values)

    # apply standard scaler
    X_test = feature_scaler.transform(X_test)
    df_test_features[df_test_features.columns] = X_test

    # important: reorder index
    df_ref = pd.read_csv("../handout/test_features.csv").set_index(["pid"])
    df_test_features.reindex(df_ref.index.values.tolist())


    print('...done')

    # Intermittent step: save the preprocessed data
    df_train_features.to_csv('df_train_features.csv')
    df_train_labels.to_csv('df_train_labels.csv')
    df_test_features.to_csv('df_test_features.csv')

else:
    print('Loading preprocessed data...')
    df_train_features = pd.read_csv('df_train_features.csv')
    df_train_labels = pd.read_csv('df_train_labels.csv')
    df_test_features = pd.read_csv('df_test_features.csv')
    df_train_features = df_train_features.set_index(["pid"]).sort_values(['pid'])
    df_train_labels = df_train_labels.set_index(["pid"]).sort_values(['pid'])
    df_test_features = df_test_features.set_index(["pid"])
    print('...done')


# ================================
# create dataframe for test labels
# ================================
df_test_labels = pd.DataFrame(index=df_test_features.index)
df_test_labels.index.names = ['pid']

# ============================
# Task 1: predict medical test
# ============================
svm = SVC(probability=True,class_weight='balanced')

# define parameter distributions
param_dist = {
    'C': scipy.stats.expon(scale=1),
    'gamma': scipy.stats.expon(scale=0.1),
    'kernel': ['rbf']}

# Perform randomized search cv to find optimal parameters
svm_search = RandomizedSearchCV(
    svm,
    param_distributions=param_dist,
    cv=2,
    n_iter=8,
    scoring="roc_auc",
    error_score=0,
    verbose=3,
    n_jobs=-1)

print('TASK1: predicting probabilities...')

# we let the linear svm fit and predict each test individually:
for label in classification_labels:
    # perform a randomized search CV
    y_train = df_train_labels[label].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        df_train_features,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        )

    svm_search.fit(X_train, y_train)

    print(svm_search.best_estimator_.predict_proba(X_test)[:, 1])
    print(
        f"ROC score on test set "
        f"{roc_auc_score(y_test, svm_search.best_estimator_.predict_proba(X_test)[:, 1])}"
    )
    print(f"CV score {svm_search.best_score_}")
    print(f"Best parameters {svm_search.best_params_}")

    joblib.dump(
        svm_search.best_estimator_,
        f"svm_search_job_{classification_labels}.pkl",
    )
    ROC_scores.append(roc_auc_score(y_test, svm_search.best_estimator_.predict_proba(X_test)[:, 1]))
    y_pred = svm_search.best_estimator_.predict_proba(df_test_features)[:, 1]
    df_test_labels[label] = y_pred

print('...done')
print(ROC_scores)


# ======================
# Task 2: predict sepsis
# ======================
clf = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
print('TASK2: predicting probabilities...')
y_train = df_train_labels[sepsis_label].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(
    df_train_features,
    y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    )
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)
# compute ROC curve and ROC area
print('compute ROC curve and ROC area...')
fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
# now, we perform the fit on the entire dataset:
clf.fit(df_train_features.to_numpy(), df_train_labels[sepsis_label].to_numpy())
print('ok')
y_pred = clf.predict_proba(df_test_features)[:, 1]
df_test_labels[sepsis_label] = y_pred
print('...done')

# ===========================
# Task 3: predict vital signs
# ===========================
en = ElasticNet()
print('TASK3: predicting regression values')

# define parameter distributions
param_dist = {
    'alpha': scipy.stats.expon(scale=0.1),
    }

# Perform randomized search cv to find optimal parameters
en_search = RandomizedSearchCV(
    en,
    param_distributions=param_dist,
    cv=4,
    n_iter=20,
    scoring="r2",
    error_score=0,
    verbose=1,
    n_jobs=-1)

# we let the svr fit and predict each vital sign individually:
for label in regression_labels:
    print(label)
    # perform a randomized search CV
    y_train = df_train_labels[label].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
    df_train_features,
    y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    )

    en_search.fit(X_train, y_train)
    y_pred = en_search.best_estimator_.predict(df_test_features)
    df_test_labels[label] = y_pred
    print(
        f"R2 score on test set "
        f"{r2_score(y_test, en_search.best_estimator_.predict(X_test))}"
    )
    print(f"CV score {en_search.best_score_}")
    print(f"Best parameters {en_search.best_params_}")
print('...done')

# suppose df is a pandas dataframe containing the result
df_test_labels.to_csv('prediction.zip', index=True, float_format='%.3f', compression='zip')
# try to evaluate the performance:
# we do 5-fold cross-validation:

# TODO: include 5-fold CV test
