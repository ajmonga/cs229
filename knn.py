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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

class_names = ['bust', 'nfl-ready']
y_true_lables = 0;
y_pred_lables = 0;

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
                     color="white" if cm[i, j] > thresh else "red")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


year = 1998;

total_size = 0;
for iter in range (0, 18):
    print iter
    train_file = "QB_stats_numerical_data_%d.csv" % (year + iter);
    test_file = "QB_stats_numerical_data_%d_test.csv" % (year + iter);
    print(train_file);
    #print(class_names)
    #Read in all the data
    with open(train_file, 'rb') as f:
        qb_stats = list(csv.reader(f, delimiter=","))
    qb_stats = np.array(qb_stats, dtype=np.int32)

    with open(test_file, 'rb') as f_test:
        qb_stats_test = list(csv.reader(f_test, delimiter=","))
    qb_stats_test = np.array(qb_stats_test, dtype=np.int32)

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

    m = qb_stats_test.shape[0]
    n = qb_stats_test.shape[1]
    #prepare the test set,
    x_test = qb_stats_test[:, 0:n-1]
    #normalize the data to have zero mean and 1 std
    x_test = scaler.transform(x_test)
    #print(x_test)
    #check the shape of the training set
    #print(x_test.shape)
    #prepare the labels, pick the last column
    y_test = qb_stats_test[:, n-1]
    #print(y_test)
    print(y_test.shape)

    #convert training data into np array to feed into sklearn
    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train)

    #convert test data into np array to feed into sklearn
    y_test = np.asarray(y_test)
    x_test = np.asarray(x_test)

    #print(x_array)

    #Apply Random forest

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #print(y_pred)
    if (iter == 0):
            y_true_lables = y_test;
            print y_test.shape[0]
            y_pred_lables = y_pred;
            total_size = y_pred.shape[0];
    else:
            total_size = total_size + y_pred.shape[0];
            print y_true_lables.shape[0]
            print y_test.shape[0]
            y_true_lables = np.concatenate((y_true_lables, y_test));
            y_pred_lables = np.concatenate((y_pred_lables, y_pred));

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true_lables, y_pred_lables)
np.set_printoptions(precision=2)

print(classification_report(y_true_lables, y_pred_lables, target_names=class_names))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

