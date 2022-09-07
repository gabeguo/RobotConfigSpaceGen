import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

import time

# CONSTANTS

TN = 0
TP = 1
FN = 2
FP = 3

N_NEIGHBORS = 5

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
    plt.scatter([x[0] for x in X_pos], [x[1] for x in X_pos], c='#ff0000', label='collision')
    plt.scatter([x[0] for x in X_neg], [x[1] for x in X_neg], c='#00ff00', label='free')

    plt.xlabel('theta1 (deg)')
    plt.ylabel('theta2 (deg)')
    plt.grid(visible=True)
    plt.title('training data')
    plt.legend()

    plt.show()

    return

def plot_results(X, Y_actual, Y_pred):
    # convert to deg
    X = rad2deg(X)

    # first show confusion matrix
    cm = confusion_matrix(Y_actual, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['free space', 'collision'])
    disp.plot()
    plt.title('confusion matrix for configuration space prediction of 2DOF arm')
    plt.show()

    # now show estimated configuration space
    confusion_array = []
    for i in range(len(X)):
        if Y_actual[i] == Y_pred[i]:
            if Y_pred[i] == 0:
                confusion_array.append(TN)
            else:
                confusion_array.append(TP)
        else:
            if Y_pred[i] == 0:
                confusion_array.append(FN)
            else:
                confusion_array.append(FP)

    X_tn = [X[i] for i in range(len(X)) if confusion_array[i] == TN]
    X_tp = [X[i] for i in range(len(X)) if confusion_array[i] == TP]
    X_fn = [X[i] for i in range(len(X)) if confusion_array[i] == FN]
    X_fp = [X[i] for i in range(len(X)) if confusion_array[i] == FP]

    plt.scatter([x[0] for x in X_tn], [x[1] for x in X_tn], c='#00ff00', label='correctly predicted free space')
    plt.scatter([x[0] for x in X_tp], [x[1] for x in X_tp], c='#0000ff', label='correctly predicted collision')
    plt.scatter([x[0] for x in X_fn], [x[1] for x in X_fn], c='#ff0000', label='wrongly predicted free space')
    plt.scatter([x[0] for x in X_fp], [x[1] for x in X_fp], c='#888888', label='wrongly predicted collision')

    plt.xlabel('theta1 (deg)')
    plt.ylabel('theta2 (deg)')
    plt.grid(visible=True)
    plt.legend()
    plt.title('configuration space predictions for 2DOF arm')

    plt.show()

    print('testing dataset size:', len(X))
    print('accuracy:', round(accuracy_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('precision:', round(precision_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('recall:', round(recall_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print('f1:', round(f1_score(y_true=Y_actual, y_pred=Y_pred), 3))
    print()

    return

def do_cross_val(X, Y, folds):
    print('Logit')
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, Y, cv=folds)
    print(scores)

    print('KNN')
    clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    scores = cross_val_score(clf, X, Y, cv=folds)
    print(scores)

def plot_errors(X, Y, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, \
        test_size=test_size, random_state=42)

    clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)

    print('training dataset size:', len(X_train))

    start = time.time()

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in fitting on', len(X_train), 'points and testing on', len(X_test), 'points:', elapsed, 'seconds')

    plot_training_data(X_train, Y_train)
    plot_results(X=X_test, Y_actual=Y_test, Y_pred=Y_pred)

    return

def main():
    X, Y = read_data()

    #plot_training_data(X, Y)
    #do_cross_val(X, Y, folds=5)

    plot_errors(X, Y, test_size=0.5)

    return

if __name__ == "__main__":
    main()
