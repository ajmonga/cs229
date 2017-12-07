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

class_names = ['bust', 'nfl-ready']

print(class_names)
#Read in all the data
with open("QB_stats_numerical_data_2004.csv", 'rb') as f:
    qb_stats = list(csv.reader(f, delimiter=","))
qb_stats = np.array(qb_stats, dtype=np.int32)

with open("QB_stats_numerical_data_2004_test.csv", 'rb') as f_test:
    qb_stats_test = list(csv.reader(f_test, delimiter=","))
qb_stats_test = np.array(qb_stats_test, dtype=np.int32)

#check the shape of the data
print(qb_stats.shape)
m = qb_stats.shape[0]
n = qb_stats.shape[1]

#prepare the training set, leave the last column
x_train = qb_stats[:, 0:n-1]
x_train_non_normalized = x_train
#normalize the data to have zero mean and 1 std
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
print(x_train)
#check the shape of the training set
print(x_train.shape)
#prepare the labels, pick the last column
y_train = qb_stats[:, n-1]
print(y_train)
print('ytrain shape')
print(y_train.shape)

m = qb_stats_test.shape[0]
n = qb_stats_test.shape[1]
#prepare the test set,
x_test = qb_stats_test[:, 0:n-1]
#normalize the data to have zero mean and 1 std
x_test = scaler.transform(x_test)
print(x_test)
#check the shape of the training set
print(x_test.shape)
#prepare the labels, pick the last column
y_test = qb_stats_test[:, n-1]
print(y_test)
print(y_test.shape)

#convert training data into np array to feed into sklearn
y_train = np.asarray(y_train)
x_train = np.asarray(x_train)

#convert test data into np array to feed into sklearn
y_test = np.asarray(y_test)
x_test = np.asarray(x_test)

#print(x_array)

#Apply SVM

for kernel in ('poly', 'rbf', 'linear'):
#for kernel in ('linear'):
    clf = SVC(kernel=kernel, gamma=2, degree=2, C=1 )
    clf.fit(x_train, y_train)
    #scoring = ['precision_macro', 'recall_macro']
    #scores = cross_validate(clf, x_train, y_train, scoring=scoring,
    #cv=5, return_train_score=False)
                            #score = cross_validation.cross_val_score(clf, x_train,y_train, cv=5, n_jobs=1).mean()
                            #print(scores)
    print(kernel)
    #clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    print(y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print(classification_report(y_test, y_pred, target_names=class_names))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

pca = PCA(n_components=3)
x_train_pca = pca.fit(x_train_non_normalized).transform(x_train_non_normalized)

plt.figure()
colors = ['navy', 'darkorange']

lw = 2

for color, i, class_names in zip(colors, [0, 1], class_names):
    plt.scatter(x_train_pca[y_train == i, 0], x_train_pca[y_train == i, 1], color=color, alpha=.8, lw=lw,
                label=class_names)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of NFL Data')



plt.show()
