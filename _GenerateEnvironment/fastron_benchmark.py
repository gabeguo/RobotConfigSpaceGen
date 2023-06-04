import pyximport
pyximport.install()

import sys
sys.path.append('../fastron_python')
import numpy as np
from fastronWrapper.fastronWrapper import PyFastron

from sklearn.metrics import confusion_matrix
import time

all_data = np.load('configs.npy')

y = np.load('labels.npy')
y = np.reshape(y, (-1, 1)).astype(float)

num_training_samples = 20000

data_train = all_data[:num_training_samples]
data_test = all_data[num_training_samples:]
y_train = y[:num_training_samples]
y_test = y[num_training_samples:]

# Initialize PyFastron
fastron = PyFastron(data_train) # where data.shape = (N, d)

fastron.y = y_train # where y.shape = (N,)

fastron.g = 10
fastron.maxUpdates = 10000
fastron.maxSupportPoints = 1500
fastron.beta = 100

start = time.time()
# Train model
fastron.updateModel()

# Predict values for a test set
pred = fastron.eval(data_test) # where data_test.shape = (N_test, d) 

end = time.time()
elapsed = round(end - start, 3)
print('time elapsed in fitting on', len(data_train), 'points and testing on', len(data_test), 'points:', elapsed, 'seconds')

cm = confusion_matrix(y_true=y_test.astype(int).flatten(), y_pred=pred.astype(int).flatten())
print(cm)

TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

# traditional accuracy, including uncertain points
print()
print('accuracy:', round((TP+TN)/(TP+TN+FP+FN), 3))
print('sensitivity (TPR):', round(TP/(TP+FN), 3))
print('specificity (TNR):', round(TN/(TN+FP), 3))