import multiple_kuka_gather_collision_points
import predict_points
import numpy as np
import json

robot_positions = [[0, -0.6, 0], [0, 0.6, 0], [-0.6, 0, 0], [0.6, 0, 0]]
robot_orientations = [[0, 0, 0], [0, 0, np.pi], [0, 0, np.pi / 2], [0, 0, -np.pi / 2]]

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

    #print(simulation_res)
    #print(prediction_res)

    full_res[num_robots] = {**simulation_res, **prediction_res}

print(json.dumps(full_res, indent=4))

with open('results.json', 'w') as fout:
    json.dump(full_res, fout)
