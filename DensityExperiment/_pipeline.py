import multiple_kuka_gather_collision_points
import predict_points
import numpy as np
import json
import random

radius = 0.4
max_height=1
obstacle_scale = 0.25
MAX_NUM_OBSTACLES = 8

obstacle_positions = list()
obstacle_orientations = list()
nums_obstacles = [j for j in range(MAX_NUM_OBSTACLES)]
#random.shuffle(nums_robots)
for i in nums_obstacles:
    theta = (i / MAX_NUM_OBSTACLES) * (2 * np.pi)
    obstacle_positions.append([radius * np.cos(theta), radius * np.sin(theta), max_height * 0.75])
    obstacle_orientations.append([0, 0, theta])
    obstacle_positions.append([radius * np.cos(theta), radius * np.sin(theta), max_height * 0.25])
    obstacle_orientations.append([0, 0, theta])

full_res = {}

for num_obstacles in range(1, 2 * MAX_NUM_OBSTACLES+1):
    curr_pos = obstacle_positions[:num_obstacles]
    curr_ori = obstacle_orientations[:num_obstacles]

    print('\n*** *** ***\n{} OBSTACLE EXPERIMENT\n*** *** ***'.format(num_obstacles))

    print('\n** data gathering **\n')
    simulation_res = multiple_kuka_gather_collision_points.main(NUM_ITERATIONS=10000, NUM_OBSTACLES=num_obstacles, \
        obstacle_positions=curr_pos, obstacle_orientations=curr_ori, obstacle_scale=obstacle_scale)
    print('\n** machine learning **\n')
    prediction_res = predict_points.main()

    full_res[num_obstacles] = {**simulation_res, **prediction_res}

print(json.dumps(full_res, indent=4))

with open('results.json', 'w') as fout:
    json.dump(full_res, fout)
