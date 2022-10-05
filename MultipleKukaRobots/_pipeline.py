import multiple_kuka_gather_collision_points
import predict_points
import numpy as np
import json

radius = 0.5

MAX_NUM_ROBOTS = 8

robot_positions = list()
robot_orientations = list()
for i in range(MAX_NUM_ROBOTS):
    theta = (i / MAX_NUM_ROBOTS) * (2 * np.pi)
    robot_positions.append([radius * np.cos(theta), radius * np.sin(theta), 0])
    robot_orientations.append([0, 0, theta])

full_res = {}

for num_robots in range(2, len(robot_positions) + 1):
    curr_pos = robot_positions[:num_robots]
    curr_ori = robot_orientations[:num_robots]

    print('\n*** *** ***\n{} ROBOT EXPERIMENT\n*** *** ***'.format(num_robots))

    print('\n** data gathering **\n')
    simulation_res = multiple_kuka_gather_collision_points.main(NUM_ITERATIONS=1000, NUM_ROBOTS=num_robots, \
        robot_positions=curr_pos, robot_orientations=curr_ori)
    print('\n** machine learning **\n')
    prediction_res = predict_points.main()

    full_res[num_robots] = {**simulation_res, **prediction_res}

print(json.dumps(full_res, indent=4))

with open('results.json', 'w') as fout:
    json.dump(full_res, fout)
