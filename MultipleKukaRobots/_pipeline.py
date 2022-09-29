import multiple_kuka_gather_collision_points
import predict_points
import numpy as np

robot_positions = [[0, -0.25, 0], [0, 0.25, 0], [-0.25, 0, 0], [0.25, 0, 0]]
robot_orientations = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

for num_robots in range(2, len(robot_positions)):
    curr_pos = robot_positions[:num_robots]
    curr_ori = robot_orientations[:num_robots]

    print('\n***\n***\nTRIAL {}\n***\n***'.format(num_robots - 1))

    print('\n**\ndata gathering\n**\n')
    multiple_kuka_gather_collision_points.main(NUM_ITERATIONS=10000, NUM_ROBOTS=num_robots, \
        robot_positions=curr_pos, robot_orientations=curr_ori)
    print('\n\n**\nmachine learning\n**\n')
    predict_points.main()
