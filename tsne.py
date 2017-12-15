print(__doc__)
import numpy as np
import itertools
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

class_names = ['bust', 'nfl-ready']


train_file = "QB_stats_numerical_data.csv";
#print(train_file);
#print(class_names)
#Read in all the data
with open(train_file, 'rb') as f:
    qb_stats = list(csv.reader(f, delimiter=","))
qb_stats = np.array(qb_stats, dtype=np.int32)

#check the shape of the data
#print(qb_stats.shape)
m = qb_stats.shape[0]
n = qb_stats.shape[1]

#prepare the training set, leave the last column
x_train = qb_stats[:, 0:n-1]
x_train_non_normalized = x_train
#normalize the data to have zero mean and 1 std
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
#print(x_train)
#check the shape of the training set
#print(x_train.shape)
#prepare the labels, pick the last column
y_train = qb_stats[:, n-1]
#print(y_train)
#print('ytrain shape')
#print(y_train.shape)

#convert training data into np array to feed into sklearn
y_train = np.asarray(y_train)
x_train = np.asarray(x_train)

X_embedded = TSNE(n_components=2).fit_transform(x_train)
X_embedded.shape





