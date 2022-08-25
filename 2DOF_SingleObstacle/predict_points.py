import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

def main():
    X, Y = read_data()

    print('Logit')
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, Y, cv=5)
    print(scores)

    print('KNN')
    clf = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(clf, X, Y, cv=5)
    print(scores)

    return

if __name__ == "__main__":
    main()
