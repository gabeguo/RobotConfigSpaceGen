# DEEP LEARNING

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNN():
    def fit(self, X_train, Y_train, learning_rate=0.01, DOF=35, BATCH_SIZE=5, EPOCHS=20):
        self.DOF = DOF
        for data_point in X_train:
            while len(data_point) < self.DOF:
                data_point.append(0)

        model = nn.Sequential(
              nn.Linear(self.DOF, 100),
              nn.Tanh(),
              nn.Linear(100, 150),
              nn.Tanh(),
              nn.Linear(150, 200),
              nn.Tanh(),
              nn.Linear(200, 250),
              nn.Tanh(),
              nn.Linear(250, 1)
            )

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.05)

        for i in range(EPOCHS):
            total_loss = 0
            # train
            for t in range(0, len(Y_train), BATCH_SIZE):
                x = torch.FloatTensor(X_train[t:t+BATCH_SIZE])
                y = torch.FloatTensor([Y_train[t:t+BATCH_SIZE]]).reshape((BATCH_SIZE, 1))

                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss
                model.zero_grad()
                loss.backward()
                optimizer.step()

        self.model = model

        return

    def predict(self, X_test):
        for data_point in X_test:
            while len(data_point) < self.DOF:
                data_point.append(0)
        return [1 if y[0] >= 0.5 else 0 for y in self.model(torch.FloatTensor(X_test)).tolist()]
