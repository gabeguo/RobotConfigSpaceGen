import multiple_kuka_gather_collision_points
import predict_points
import numpy as np
import json
import random

full_res = dict()

radius = 0.5
MAX_NUM_ROBOTS = 6
robot_positions = list()
robot_orientations = list()
nums_robots = [j for j in range(MAX_NUM_ROBOTS)]
#random.shuffle(nums_robots)
for i in nums_robots:
    theta = (i / MAX_NUM_ROBOTS) * (2 * np.pi)
    robot_positions.append([radius * np.cos(theta), radius * np.sin(theta), 0])
    robot_orientations.append([0, 0, theta])

simulation_res = multiple_kuka_gather_collision_points.main(NUM_ITERATIONS=5 * 1000 * 1000, NUM_ROBOTS=MAX_NUM_ROBOTS, \
    robot_positions=robot_positions, robot_orientations=robot_orientations)

for test_size in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    print('\n\n*** test size = {} ***\n\n'.format(test_size))
    prediction_res = predict_points.main(test_size=test_size)
    full_res[test_size] = {**simulation_res, **prediction_res}

print(json.dumps(full_res, indent=4))

with open('sample_percent_results.json', 'w') as fout:
    json.dump(full_res, fout)
