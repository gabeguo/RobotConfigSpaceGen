import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import time

import math

# CONSTANTS

TN = 0
TP = 1
FN = 2
FP = 3

N_NEIGHBORS = 5
UNCERTAIN = -1

# METHODS

def read_data():
    FILENAME = 'collision_data.csv'
    X, Y = [], []
    with open(FILENAME, 'r') as input:
        reader = csv.reader(input)
        next(reader)

        for row in reader:
            X.append([float(item) for item in row[:-1]])
            Y.append(int(row[-1]))
    return X, Y

def rad2deg(X):
    return [[the_theta / np.pi * 180 for the_theta in x] for x in X]

def plot_results(X, Y_actual, Y_confidence, Y_pred):
    Y_pred_with_uncertain = [1 if math.isclose(y, 1) \
                    else 0 if math.isclose(y, 0) \
                    else UNCERTAIN \
                    for y in Y_confidence]

    # first show confusion matrix
    cm = confusion_matrix(Y_actual, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['free space', 'collision'])
    disp.plot()
    plt.title('confusion matrix for configuration space prediction of 3DOF arm')
    plt.show()

    print('testing dataset size:', len(X))

    # traditional accuracy, including uncertain points
    print()
    print('accuracy with all points:', round(accuracy_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('precision excluding uncertain points:', round(precision_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('recall excluding uncertain points:', round(recall_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('f1 excluding uncertain points:', round(f1_score(y_true=Y_actual, y_pred=Y_pred), 3))

    # accuracy, EXCLUDING uncertain points (only have points we are confident in)
    certain_indices = [i for i in range(len(Y_pred_with_uncertain)) if Y_pred_with_uncertain[i] != UNCERTAIN]
    certain_y_pred = [Y_pred[i] for i in certain_indices]
    certain_y_true = [Y_actual[i] for i in certain_indices]
    print()
    print('accuracy excluding uncertain points:', round(accuracy_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    print('precision excluding uncertain points:', round(precision_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    print('recall excluding uncertain points:', round(recall_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    print('f1 excluding uncertain points:', round(f1_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    # Count number of uncertain points
    print('\nproportion of points that model is uncertain about:', round(1 - len(certain_indices) / len(Y_actual), 3))
    print()

    return

# returns training dataset with same numbers of positive and negative labels
def resample_training_data_balanced(X_train, Y_train):
    pos_indices = []
    neg_indices = []
    for i in range(len(Y_train)):
        if Y_train[i] == 1:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    num_samples_needed_per_class = min(len(pos_indices), len(neg_indices))

    new_X_train, new_Y_train = [], []
    for i in pos_indices[:num_samples_needed_per_class]:
        new_X_train.append(X_train[i])
        new_Y_train.append(Y_train[i])
    for i in neg_indices[:num_samples_needed_per_class]:
        new_X_train.append(X_train[i])
        new_Y_train.append(Y_train[i])

    return new_X_train, new_Y_train

def evaluate(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
        test_size=test_size, random_state=42)

    clf = KNeighborsRegressor(n_neighbors=N_NEIGHBORS, weights='distance')

    print('n neighbors:', N_NEIGHBORS)
    print('training dataset size:', len(X_train))

    start = time.time()

    #X_train, Y_train = resample_training_data_balanced(X_train, Y_train)

    clf.fit(X_train, Y_train)
    Y_confidence_score = clf.predict(X_test)
    Y_pred = [int(y + 0.5) for y in Y_confidence_score]

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in fitting on', len(X_train), 'points and testing on', len(X_test), 'points:', elapsed, 'seconds')

    # get dummy results
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, Y_train)
    print('dummy results:', round(accuracy_score(y_true=Y_test, y_pred=dummy.predict(X_test)), 3))

    # get accuracy
    plot_results(X=X_test, Y_actual=Y_test, Y_confidence=Y_confidence_score, Y_pred=Y_pred)

    return

def main():
    X, Y = read_data()

    evaluate(X, Y, test_size=0.8)

    return

if __name__ == "__main__":
    main()
