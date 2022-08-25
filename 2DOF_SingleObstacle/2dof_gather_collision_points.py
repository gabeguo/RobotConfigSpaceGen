# Adapted from https://github.com/adamheins/pyb_utils

#!/usr/bin/env python
"""Example demonstrating collision detection and shortest distance queries."""
import numpy as np
import pybullet as pyb
import pybullet_data

from collision_detection import NamedCollisionObject, CollisionDetector
from itertools import combinations

import csv

def load_environment(client_id):
    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id
    )

    # KUKA iiwa robot arm
    arm_id = pyb.loadURDF(
        "./basic_arm.urdf",#"kuka_iiwa/model.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=1
    )

    # some cubes for obstacles
    cube1_id = pyb.loadURDF(
        "cube.urdf", [1, 1, 0.5], useFixedBase=True, physicsClientId=client_id
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot": arm_id,
        "ground": ground_id,
        "cube1": cube1_id,
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

    # main simulation server, with a GUI
    sim_id = pyb.connect(pyb.DIRECT)

    collision_bodies = load_environment(sim_id)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking
    ground = NamedCollisionObject("ground")
    cube1 = NamedCollisionObject("cube1")
    link1 = NamedCollisionObject("robot", "link1")
    link2 = NamedCollisionObject("robot", "link2")
    link3 = NamedCollisionObject("robot", "link3")
    collision_objects = [ground, cube1, link1, link2, link3]

    collision_pairs = [(link3, cube1), (link1, link3), (ground, link3)]
    #collision_pairs = list(combinations(collision_objects, 2))

    col_detector = CollisionDetector(
        sim_id,
        collision_bodies,
        collision_pairs
    )

    COLLISION_DATA_LABELS = ['theta1', 'theta2', 'collision']
    _collision_data = []

    NUM_ITERATIONS = 50
    for i in range(0, NUM_ITERATIONS):

        # compute shortest distances for a configuration
        FIXED_JOINT_ANGLE = np.pi / 4
        MAX_JOINT_ANGLE = np.pi * 3/4
        q = [FIXED_JOINT_ANGLE, MAX_JOINT_ANGLE * np.random.random(), MAX_JOINT_ANGLE * np.random.random()]
        distances = col_detector.compute_distances(q, max_distance=20)
        in_col = (distances < 0).any()

        _collision_data.append([q[1], q[2], int(in_col)])

    write_collision_data(COLLISION_DATA_LABELS, _collision_data)
    return

if __name__ == "__main__":
    main()
