# Adapted from https://github.com/adamheins/pyb_utils

#!/usr/bin/env python
"""Example demonstrating collision detection and shortest distance queries."""
import numpy as np
import pybullet as pyb
import pybullet_data

from collision_detection import NamedCollisionObject, CollisionDetector
from itertools import combinations

import csv

import time

def load_environment(client_id, NUM_ROBOTS, robot_positions, robot_orientations):
    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    arm_id = [None for i in range(NUM_ROBOTS)]

    for i in range(NUM_ROBOTS):
        arm_id[i] = pyb.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=robot_positions[i],
            baseOrientation=pyb.getQuaternionFromEuler(robot_orientations[i]),
            useFixedBase=True,
            physicsClientId=client_id,
            globalScaling=2
        )

    # cube1_id = pyb.loadURDF(
    #     "r2d2.urdf", [1.1, 1.25, 1.1], \
    #     baseOrientation=pyb.getQuaternionFromEuler([np.pi / 4, np.pi / 3, np.pi / 5]), \
    #     useFixedBase=True, globalScaling=1.5, physicsClientId=client_id
    # )
    # cube2_id = pyb.loadURDF(
    #     "duck_vhacd.urdf", [-0.9, -1.25, 1.5], \
    #     baseOrientation=pyb.getQuaternionFromEuler([np.pi / 6, np.pi / 7, np.pi / 9]), \
    #     useFixedBase=True, globalScaling=7, physicsClientId=client_id
    # )
    # cube3_id = pyb.loadURDF(
    #     "teddy_vhacd.urdf", [0.85, -0.75, 2], \
    #     baseOrientation=pyb.getQuaternionFromEuler([np.pi * 2/3, np.pi * 4/5, np.pi * 5/8]), \
    #     useFixedBase=True, globalScaling=4, physicsClientId=client_id
    # )
    # cube4_id = pyb.loadURDF(
    #     "cube.urdf", [0.75, 1, 1.25], \
    #     baseOrientation=pyb.getQuaternionFromEuler([np.pi * 1/2, np.pi * 2/3, np.pi * 1/4]), \
    #     useFixedBase=True, globalScaling=0.5, physicsClientId=client_id
    # )
    # cube5_id = pyb.loadURDF(
    #     "soccerball.urdf", [1, -1.5, 0.75], \
    #     baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]), \
    #     useFixedBase=True, globalScaling=0.5, physicsClientId=client_id
    # )

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot{}".format(i) : arm_id[i] for i in range(NUM_ROBOTS)
    }
    # {
    #     "cube1": cube1_id,
    #     "cube2": cube2_id,
    #     "cube3": cube3_id,
    #     "cube4": cube4_id,
    #     "cube5": cube5_id,
    # }
    return bodies

def write_collision_data(fields, data):
    assert len(fields) == len(data[0])
    with open('collision_data.csv', 'w') as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(data)
    return

def main(NUM_ITERATIONS = 10000, NUM_ROBOTS = 4, \
    robot_positions=[[0, -0.25, 0], [0, 0.25, 0], [-0.25, 0], [0.25, 0, 0]], \
    robot_orientations=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]):

    # main simulation server
    sim_id = pyb.connect(pyb.DIRECT)

    collision_bodies = load_environment(sim_id, NUM_ROBOTS, robot_positions, robot_orientations)

    for body in collision_bodies:
        #print(pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id))
        is_mesh = pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id)[0][2] == pyb.GEOM_MESH
        print(body, 'is mesh:', is_mesh)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking

    obstacles = [] #[NamedCollisionObject("cube{}".format(i)) for i in range(1,5+1)]
    robotlinks = [\
        [NamedCollisionObject("robot{}".format(i), "lbr_iiwa_link_{}".format(j)) \
                    for j in range(1, 7+1)] + \
        [NamedCollisionObject("robot{}".format(i), None)] \
                    for i in range(NUM_ROBOTS)]

    collision_objects = [the_link for the_robotlinks in robotlinks for the_link in the_robotlinks] \
        + obstacles

    collision_pairs = [(link_i, link_j) \
            for i in range(len(robotlinks)) \
                for j in range(i + 1, len(robotlinks)) \
                    for link_i in robotlinks[i] \
                        for link_j in robotlinks[j]] + \
        [(the_link, the_obstacle) \
            for the_robotlinks in robotlinks \
                for the_link in the_robotlinks \
                    for the_obstacle in obstacles]

    #print(collision_pairs)

    col_detector = CollisionDetector(
        sim_id,
        collision_bodies,
        collision_pairs
    )

    COLLISION_DATA_LABELS = \
        ['robot{}_theta{}'.format(i, j) \
            for i in range(NUM_ROBOTS) \
            for j in range(1, 7+1)] + \
        ['collision']

    _collision_data = []

    MAX_JOINT_ANGLE = [theta * np.pi / 180 for theta in [170, 120, 170, 120, 170, 120, 175]]
    # joint limits: https://www.researchgate.net/figure/Joint-limits-of-KUKA-LBR-iiwa-14-R820-45_tbl1_339394448

    # generate angles (since it is biased to generate them while also detecting if they collide)
    start = time.time()

    Q_trial_robot = [] # Q_trial_robot[i][j] = configuration of robot j on trial i
    for i in range(0, NUM_ITERATIONS):
        Q_trial_robot.append(list())
        for j in range(NUM_ROBOTS):
            Q_trial_robot[i].append(list())
            for dof in range(7):
                Q_trial_robot[i][j].append(MAX_JOINT_ANGLE[dof] * 2 * np.random.random() - MAX_JOINT_ANGLE[dof])

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in generating', NUM_ITERATIONS, 'configurations:', elapsed, 'seconds')

    Q = [[theta for robot_data in trial for theta in robot_data] for trial in Q_trial_robot]

    # start detecting collisions
    start = time.time()

    for i in range(0, NUM_ITERATIONS):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances_multi_robot(Q_trial_robot[i], max_distance=0)
        in_col = (distances < 0).any()

        Q[i].append(int(in_col))
        _collision_data.append(Q[i])

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in checking', NUM_ITERATIONS, 'configurations for collision:', elapsed, 'seconds')
    # stop detecting collisions

    write_collision_data(COLLISION_DATA_LABELS, _collision_data)

    # GUI dummy demo of simulation starting point; does not move
    # gui_id = pyb.connect(pyb.GUI)
    # gui_collision_bodies = load_environment(gui_id, NUM_ROBOTS, robot_positions, robot_orientations)
    # pyb.resetDebugVisualizerCamera( cameraDistance=5, cameraYaw=10, cameraPitch=-40, cameraTargetPosition=[0, 0, 0], \
    #     physicsClientId=gui_id)
    # gui_col_detector = CollisionDetector(gui_id, gui_collision_bodies, collision_pairs)
    # for i in range(10):
    #     q_curr = [[MAX_JOINT_ANGLE[j] * 2 * np.random.random() - MAX_JOINT_ANGLE[j] for j in range(7)] \
    #         for k in range(NUM_ROBOTS)]
    #     gui_col_detector.compute_distances_multi_robot(q_curr, max_distance=0)
    #     input()
    #     pyb.stepSimulation(physicsClientId=gui_id)

    # cleanup
    # pyb.disconnect(physicsClientId=gui_id)
    pyb.disconnect(physicsClientId=sim_id)

    return

if __name__ == "__main__":
    main()
