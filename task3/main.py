"""
Task 3: Protein Mutation Classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021

INTRODUCTION

Proteins are large molecules. Their blueprints are encoded in the DNA of
biological organisms. Each protein consists of many amino acids: for example,
our protein of interest consists of a little less than 400 amino acids. Once the
protein is created (synthesized), it folds into a 3D structure, which can be
seen in Figure 1. The mutations influence what amino acids make up the protein,
and hence have an effect on its shape.

TASK

The goal of this task is to classify mutations of a human antibody protein into
active (1) and inactive (0) based on the provided mutation information. Under
active mutations the protein retains its original function, and inactive
mutation cause the protein to lose its function. The mutations differ from each
other by 4 amino acids in 4 respective sites. The sites or locations of the
mutations are fixed. The amino acids at the 4 mutation sites are given as
4-letter combinations, where each letter denotes the amino acid at the
corresponding mutation site. Amino acids at other places are kept the same and
are not provided.

For example, FCDI corresponds to amino acid F (Phenylanine) being in the first
site, amino acid C (Cysteine) being in the second site and so on. The Figure 2
gives translation from symbols to amino acid chemical names for the interested
students. The biological and chemical aspects can be abstracted to solve this
task.

EVALUATION

For the practical purposes, it is very important to detect nearly all active
mutations such that they can be evaluated. Hence we need to maximize recall
(true positive rate), but at the same time we want to have equally good
precision. Therefore, we use F1 score which captures both precision and recall.
"""

import time as timing
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.002

# set random seed
np.random.seed(42)
torch.manual_seed(42)


class AverageMeter:
    """ Computes and stores the average and current value. """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# training data loaders
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# test data loader
class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


# binary classification neural network
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.classifier1 = nn.Sequential(

            nn.Linear(len(X_test[0]), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1)
        )
        self.classifier2 = nn.Sequential(

            nn.Linear(len(X_test[0]), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        return self.classifier2(inputs)


# manual f1 score computation
def f1_acc(y_pred, y_true):
    tp = (y_true * y_pred).sum().float()
    tn = ((1 - y_true) * (1 - y_pred)).sum().float()
    fp = ((1 - y_true) * y_pred).sum().float()
    fn = (y_true * (1 - y_pred)).sum().float()
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


# -----------------------------------------------------------------------------
# preprocessing
df_train = pd.read_csv("/handout/train.csv")
df_test = pd.read_csv("/handout/test.csv")

# # data is very imbalanced
# sns.countplot(x = 'Active', data=df_train)
# plt.show()

# split strings of amino acids, maintaining the sites
X_train = df_train['Sequence'].values
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = df_test['Sequence'].values
X_test = [list(X_test[i]) for i in range(len(X_test))]

# get activity label of mutations
y_train = df_train['Active'].values

# encode data with one hot encoding, preserve the order of the mutation
enc = OneHotEncoder()
enc.fit(X_train)
X_train = enc.transform(X_train).toarray()
X_test = enc.transform(X_test).toarray()
print(enc.categories_)

# scale data
# scaler = StandardScaler()
scaler = PowerTransformer()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# network architecture

# set device for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
model = BinaryClassification().to(device)
print(model)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# define loss function
# consider re-weighting the classes due to heavily imbalanced dataset
pos_frac = np.sum(y_train)/len(y_train)
pos_weight = (1-pos_frac)/pos_frac
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

# transform data into tensors for pytorch
train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = TestData(torch.FloatTensor(X_test))

# -----------------------------------------------------------------------------
# training loop
model.train()
train_results = {}

# iterate over all epochs
for epoch in range(1, EPOCHS+1):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()

    # load data per epoch
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # iterate over training mini-batches
    for i, data, in enumerate(train_loader, 1):
        # accounting
        end = timing.time()
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        bs = features.size(0)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propagation
        prediction = model(features)
        # compute the loss
        loss = criterion(prediction, labels.unsqueeze(1))
        # backward propagation
        loss.backward()
        # optimization step
        optimizer.step()

        # reports per mini-batch
        # print(classification_report(torch.round(torch.sigmoid(prediction)).detach().numpy(), labels.unsqueeze(1)))
        # print(confusion_matrix(torch.round(torch.sigmoid(prediction)).detach().numpy(), labels.unsqueeze(1)))

        # accounting
        acc = f1_acc(torch.round(torch.sigmoid(prediction)), labels.unsqueeze(1))
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(timing.time() - end)

    print(f'Epoch {epoch}. [Train] \t Time {time_.sum:.2f} Loss {loss_.avg:.2f} \t Accuracy {acc_.avg:.2f}')
    train_results[epoch] = (loss_.avg, acc_.avg, time_.avg)

# plot training process
# training = list()
# for key, values in train_results.items():
#     training.append([key, values[0], values[1], values[2]])
# training = list(map(list, zip(*training)))

# fig, axs = plt.subplots(2)
# fig.suptitle('Loss and accuracy per epoch')
# axs[0].plot(training[0], training[1], 'b')
# axs[0].set_ylabel('loss')
# axs[0].grid()
# axs[1].plot(training[0], training[2], 'b')
# axs[1].set_ylabel('accuracy')
# axs[1].set_xlabel('epoch')
# axs[1].grid()
# plt.show()

# -----------------------------------------------------------------------------
# perform predictions
model.eval()
y_pred_list = list()
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
for data in test_loader:
    with torch.no_grad():
        features = data.to(device)
        prediction = model(features)
        y_pred = torch.round(torch.sigmoid(prediction))
        y_pred_list.append(y_pred.cpu().numpy())

# save predictions to csv file
y_pred_list = [batch.squeeze().tolist() for batch in y_pred_list]
y_pred_list = np.concatenate(y_pred_list).ravel().tolist()
np.savetxt("predictions.csv", y_pred_list, fmt="%i")
