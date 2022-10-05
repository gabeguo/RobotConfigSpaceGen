# DEEP LEARNING

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNN():
    def fit(self, X_train, Y_train, learning_rate=0.01, DOF=35, BATCH_SIZE=5, EPOCHS=30):
        self.DOF = DOF
        for data_point in X_train:
            while len(data_point) < self.DOF:
                data_point.append(0)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.05)

        for i in range(EPOCHS):
            total_loss = 0
            # train
            for t in range(0, len(Y_train), BATCH_SIZE):
                curr_batch_size = min(BATCH_SIZE, len(Y_train) - t)
                x = torch.tensor(X_train[t:t+curr_batch_size], device=self.device, dtype=torch.float32)
                y = torch.tensor([Y_train[t:t+curr_batch_size]], device=self.device, dtype=torch.float32).reshape((curr_batch_size, 1))

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
        return [1 if y[0] >= 0.5 else 0 for y in self.model(torch.tensor(X_test, device=self.device, dtype=torch.float32)).tolist()]
