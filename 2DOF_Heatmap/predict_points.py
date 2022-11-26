import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import numpy as np
import math
import xgboost as xgb

import time

# CONSTANTS

TN = 0
TP = 1
FN = 2
FP = 3

N_NEIGHBORS = 7
MARGIN_OF_CERTAINTY = 1e-2#1 / N_NEIGHBORS - 1e-3 # uncertain points have scores within [MARGIN, 1 - MARGIN]
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
    return [[x[0] / np.pi * 180, x[1] / np.pi * 180] for x in X]

def plot_training_data(X, Y):
    X = rad2deg(X)
    X_pos = []
    X_neg = []
    for i in range(len(X)):
        if Y[i] == 1:
            X_pos.append(X[i])
        else:
            X_neg.append(X[i])
    plt.scatter([x[0] for x in X_pos], [x[1] for x in X_pos], s=3, c='#ff0000', label='collision')
    plt.scatter([x[0] for x in X_neg], [x[1] for x in X_neg], s=3, c='#0000ff', label='free')

    plt.xlabel('theta1 (deg)')
    plt.ylabel('theta2 (deg)')
    plt.grid(visible=True)
    plt.title('training data')
    plt.legend()

    plt.show()

    return

def plot_results(X, Y_actual, Y_confidence, Y_pred):
    Y_pred_with_uncertain = [1 if y >= 1 - MARGIN_OF_CERTAINTY \
                    else 0 if y <= MARGIN_OF_CERTAINTY \
                    else UNCERTAIN \
                    for y in Y_confidence]

    # convert to deg
    X = rad2deg(X)

    plt.scatter([x[0] for x in X], [x[1] for x in X], c=Y_confidence, s=3, alpha=1, cmap='coolwarm')

    plt.xlabel('theta1 (deg)')
    plt.ylabel('theta2 (deg)')
    plt.grid(visible=True)
    #plt.legend()
    plt.colorbar(label='probability of collision')
    plt.title('configuration space prediction heatmap for 2DOF arm')

    plt.show()

    print('\ntesting dataset size:', len(X))

    # traditional accuracy, including uncertain points
    print('\naccuracy with all points:', round(accuracy_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('precision with all points:', round(precision_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('recall with all points:', round(recall_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('f1 with all points:', round(f1_score(y_true=Y_actual, y_pred=Y_pred), 3))
    # accuracy, EXCLUDING uncertain points (only have points we are confident in)
    certain_indices = [i for i in range(len(Y_pred_with_uncertain)) if Y_pred_with_uncertain[i] != UNCERTAIN]
    certain_y_pred = [Y_pred[i] for i in certain_indices]
    certain_y_true = [Y_actual[i] for i in certain_indices]
    print('\naccuracy excluding uncertain points:', round(accuracy_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    print('precision excluding uncertain points:', round(precision_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    print('recall excluding uncertain points:', round(recall_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    print('f1 excluding uncertain points:', round(f1_score(y_true=certain_y_true, y_pred=certain_y_pred), 3))
    # Count number of uncertain points
    print('\nproportion of points that model is uncertain about:', round(1 - len(certain_indices) / len(Y_actual), 3))
    print('number of points model is certain about:', len(certain_indices))
    print()

    return

def evaluate(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
        test_size=test_size, random_state=42)

    #clf = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
    clf = xgb.XGBRegressor(booster='gbtree', \
        n_estimators=100, \
        tree_method='hist', \
        eta=0.1, \
        max_depth=10, \
        objective="binary:logistic", \
        random_state=1)

    print('training dataset size:', len(X_train))

    start = time.time()

    clf.fit(X_train, Y_train)
    Y_confidence_score = clf.predict(X_test)
    Y_pred = [int(y + 0.5) for y in Y_confidence_score]

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in fitting on', len(X_train), 'points and testing on', len(X_test), 'points:', elapsed, 'seconds')

    # get dummy results
    dummy = DummyClassifier(strategy="most_frequent")
    start = time.time()
    dummy.fit(X_train, Y_train)
    dummy_scores = dummy.predict(X_test)
    end = time.time()
    elapsed = round(end - start, 3)
    print('\ntime elapsed with dummy', len(X_train), 'points and testing on', len(X_test), 'points:', elapsed, 'seconds')
    print('dummy results:', round(accuracy_score(y_true=Y_test, y_pred=dummy.predict(X_test)), 3))

    # get accuracy
    plot_training_data(X_train, Y_train)
    plot_results(X=X_test, Y_actual=Y_test, Y_confidence=Y_confidence_score, Y_pred=Y_pred)

    return

def main():
    X, Y = read_data()

    evaluate(X, Y, test_size=0.8)

    return

if __name__ == "__main__":
    main()
