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

def load_environment(client_id):
    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    arm_id = pyb.loadURDF(
        "TwoJointRobot_wo_fixedJoints.urdf",
        basePosition=[0, 0, 0],
        baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=1
    )
    cube1_id = pyb.loadURDF(
        "r2d2.urdf", [0.1, 0.75, 0.1], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi / 4, np.pi / 3, np.pi / 5]), \
        useFixedBase=True, globalScaling=1, physicsClientId=client_id
    )
    cube2_id = pyb.loadURDF(
        "duck_vhacd.urdf", [-0.9, -0.45, -0.1], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi / 6, np.pi / 7, np.pi / 9]), \
        useFixedBase=True, globalScaling=5, physicsClientId=client_id
    )
    cube3_id = pyb.loadURDF(
        "teddy_vhacd.urdf", [0.25, -0.5, 0], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi * 2/3, np.pi * 4/5, np.pi * 5/8]), \
        useFixedBase=True, globalScaling=5, physicsClientId=client_id
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot": arm_id,
        "cube1": cube1_id,
        "cube2": cube2_id,
        "cube3": cube3_id,
    }
    return bodies

def write_collision_data(fields, data):
    assert len(fields) == len(data[0])
    with open('collision_data.csv', 'w') as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(data)
    return

def main():

    # main simulation server
    sim_id = pyb.connect(pyb.DIRECT)

    collision_bodies = load_environment(sim_id)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking
    cube1 = NamedCollisionObject("cube1")
    cube2 = NamedCollisionObject("cube2")
    cube3 = NamedCollisionObject("cube3")
    link1 = NamedCollisionObject("robot", "link_1")
    link2 = NamedCollisionObject("robot", "link_2")
    collision_objects = [cube1, cube2, cube3, link1, link2]

    collision_pairs = [(link2, cube1), (link2, cube2), (link2, cube3),\
        (link2, cube1), (link2, cube2), (link2, cube3)]

    col_detector = CollisionDetector(
        sim_id,
        collision_bodies,
        collision_pairs
    )

    COLLISION_DATA_LABELS = ['theta1', 'theta2', 'collision']
    _collision_data = []

    NUM_ITERATIONS = 100000

    start = time.time()

    for i in range(0, NUM_ITERATIONS):

        # compute shortest distances for a configuration
        FIXED_JOINT_ANGLE = np.pi / 4
        MAX_JOINT_ANGLE = np.pi * 15/16
        q = [MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE, \
            MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE, \
            FIXED_JOINT_ANGLE]
        distances = col_detector.compute_distances(q, max_distance=20)
        in_col = (distances < 0).any()

        _collision_data.append([q[0], q[1], int(in_col)])

    end = time.time()
    elapsed = round(end - start, 3)
    print('\n***\ntime elapsed in sampling', NUM_ITERATIONS, 'points:', elapsed, 'seconds\n***')

    write_collision_data(COLLISION_DATA_LABELS, _collision_data)

    # GUI dummy demo of simulation starting point; does not move
    gui_id = pyb.connect(pyb.GUI)
    gui_collision_bodies = load_environment(gui_id)
    gui_col_detector = CollisionDetector(gui_id, gui_collision_bodies, collision_pairs)
    for i in range(10):
        q = [MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE, \
            MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE, \
            FIXED_JOINT_ANGLE]
        gui_col_detector.compute_distances(q, max_distance=20)
        input()
        pyb.stepSimulation(physicsClientId=gui_id)


    return

if __name__ == "__main__":
    main()
