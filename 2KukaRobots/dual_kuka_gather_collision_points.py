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

    arm0_id = pyb.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=2
    )

    arm1_id = pyb.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 1, 0],
        baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=2
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot0": arm0_id,
        "robot1": arm1_id,
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

    for body in collision_bodies:
        #print(pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id))
        is_mesh = pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id)[0][2] == pyb.GEOM_MESH
        print(body, 'is mesh:', is_mesh)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking

    robot0links = [NamedCollisionObject("robot0", "lbr_iiwa_link_{}".format(i)) for i in range(1,7+1)]
    robot1links = [NamedCollisionObject("robot1", "lbr_iiwa_link_{}".format(i)) for i in range(1,7+1)]

    collision_objects = robot0links + robot1links

    collision_pairs = [(link0, link1) \
        for link0 in robot0links \
        for link1 in robot1links]# + \
        #list(combinations(robot0links, 2)) + list(combinations(robot1links, 2)) # need self-touching

    col_detector = CollisionDetector(
        sim_id,
        collision_bodies,
        collision_pairs
    )

    COLLISION_DATA_LABELS = \
        ['robotA_theta{}'.format(i) for i in range(1, 7+1)] + \
        ['robotB_theta{}'.format(i) for i in range(1, 7+1)] + \
        ['collision']
    _collision_data = []

    NUM_ITERATIONS = 100000
    MAX_JOINT_ANGLE = np.pi * 15/16

    # generate angles (since it is biased to generate them while also detecting if they collide)
    start = time.time()

    Q0 = []
    Q1 = []
    for i in range(0, NUM_ITERATIONS):
        q0 = []
        q1 = []
        for j in range(7):
            q0.append(MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE)
            q1.append(MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE)
        Q0.append(q0)
        Q1.append(q1)

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in generating', NUM_ITERATIONS, 'configurations:', elapsed, 'seconds')

    Q = [Q0[i] + Q1[i] for i in range(len(Q0))]

    # start detecting collisions
    start = time.time()

    for i in range(0, NUM_ITERATIONS):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances_multi_robot((Q0[i], Q1[i]), max_distance=0)
        in_col = (distances < 0).any()

        Q[i].append(int(in_col))
        _collision_data.append(Q[i])

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in checking', NUM_ITERATIONS, 'configurations for collision:', elapsed, 'seconds')
    # stop detecting collisions

    write_collision_data(COLLISION_DATA_LABELS, _collision_data)

    # GUI dummy demo of simulation starting point; does not move
    gui_id = pyb.connect(pyb.GUI)
    gui_collision_bodies = load_environment(gui_id)
    gui_col_detector = CollisionDetector(gui_id, gui_collision_bodies, collision_pairs)
    for i in range(10):
        q0 = [MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE for j in range(7)]
        q1 = [MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE for j in range(7)]
        gui_col_detector.compute_distances_multi_robot((q0, q1), max_distance=0)
        input()
        pyb.stepSimulation(physicsClientId=gui_id)


    return

if __name__ == "__main__":
    main()
