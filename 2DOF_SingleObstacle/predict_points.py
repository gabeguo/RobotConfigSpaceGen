import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

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

def plot_data(X, Y):
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
    plt.legend()

    plt.show()

    return

def main():
    X, Y = read_data()

    plot_data(X, Y)

    print('Logit')
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, Y, cv=2)
    print(scores)

    print('KNN')
    clf = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(clf, X, Y, cv=2)
    print(scores)

    return

if __name__ == "__main__":
    main()
