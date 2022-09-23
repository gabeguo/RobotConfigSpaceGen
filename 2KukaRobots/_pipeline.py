import dual_kuka_gather_collision_points
import predict_points
import numpy as np

robot_positions_list=[\
    [[0.75, -0.75, 0], [0, 0, 1.5]], \
    [[0, -0.25, 0], [0, 0.25, 0]], \
    [[0, -0.75, 0], [0, 0.75, 0]]
]
robot_orientations_list=[\
    [[0, 0, 0], [np.pi/2, np.pi/2, np.pi / 4]], \
    [[0, 0, 0], [0, 0, 0]], \
    [[0, 0, 0], [0, 0, 0]], \
]

for i in range(len(robot_positions_list)):
    curr_pos = robot_positions_list[i]
    curr_ori = robot_orientations_list[i]

    print('\n***\n***\nTRIAL {}\n***\n***'.format(i))

    print('\n**\ndata gathering\n**\n')
    dual_kuka_gather_collision_points.main(NUM_ITERATIONS=40000, robot_positions=curr_pos, robot_orientations=curr_ori)
    print('\n\n**\nmachine learning\n**\n')
    predict_points.main()
