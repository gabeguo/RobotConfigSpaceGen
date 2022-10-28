import predict_points
import numpy as np
import json
import random

full_res = dict()

for test_size in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    prediction_res = predict_points.main(test_size=test_size)
    full_res[test_size] = prediction_res

print(json.dumps(full_res, indent=4))

with open('sample_percent_results.json', 'w') as fout:
    json.dump(full_res, fout)
