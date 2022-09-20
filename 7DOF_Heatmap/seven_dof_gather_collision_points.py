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
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=2
    )
    cube1_id = pyb.loadURDF(
        "r2d2.urdf", [0.1, 1.25, 1], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi / 4, np.pi / 3, np.pi / 5]), \
        useFixedBase=True, globalScaling=1.5, physicsClientId=client_id
    )
    cube2_id = pyb.loadURDF(
        "duck_vhacd.urdf", [-0.9, -1.25, 1.5], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi / 6, np.pi / 7, np.pi / 9]), \
        useFixedBase=True, globalScaling=7, physicsClientId=client_id
    )
    cube3_id = pyb.loadURDF(
        "teddy_vhacd.urdf", [0.75, -0.5, 2], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi * 2/3, np.pi * 4/5, np.pi * 5/8]), \
        useFixedBase=True, globalScaling=4, physicsClientId=client_id
    )
    cube4_id = pyb.loadURDF(
        "cube.urdf", [0.5, 0, 2.5], \
        baseOrientation=pyb.getQuaternionFromEuler([np.pi * 1/2, np.pi * 2/3, np.pi * 1/4]), \
        useFixedBase=True, globalScaling=0.5, physicsClientId=client_id
    )
    cube5_id = pyb.loadURDF(
        "soccerball.urdf", [0, -0.5, 0], \
        baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]), \
        useFixedBase=True, globalScaling=0.5, physicsClientId=client_id
    )


    # store body indices in a dict with more convenient key names
    bodies = {
        "robot": arm_id,
        "cube1": cube1_id,
        "cube2": cube2_id,
        "cube3": cube3_id,
        "cube4": cube4_id,
        "cube5": cube5_id,
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

    obstacles = [NamedCollisionObject("cube{}".format(i)) for i in range(1,5+1)]
    links = [NamedCollisionObject("robot", "lbr_iiwa_link_{}".format(i)) for i in range(1,7+1)]

    collision_objects = obstacles + links

    collision_pairs = [(the_link, the_obstacle) \
        for the_link in links \
        for the_obstacle in obstacles] + \
        [] # need self-touching

    col_detector = CollisionDetector(
        sim_id,
        collision_bodies,
        collision_pairs
    )

    COLLISION_DATA_LABELS = ['theta{}'.format(i) for i in range(1, 7+1)] + ['collision']
    _collision_data = []

    NUM_ITERATIONS = 10000
    MAX_JOINT_ANGLE = np.pi * 15/16

    # generate angles (since it is biased to generate them while also detecting if they collide)
    start = time.time()

    Q = []
    for i in range(0, NUM_ITERATIONS):
        q = []
        for j in range(7):
            q.append(MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE)
        Q.append(q)

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in generating', NUM_ITERATIONS, 'configurations:', elapsed, 'seconds')

    # start detecting collisions
    start = time.time()

    for i in range(0, NUM_ITERATIONS):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances(Q[i], max_distance=0)
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
        q = [MAX_JOINT_ANGLE * 2 * np.random.random() - MAX_JOINT_ANGLE for j in range(7)]
        gui_col_detector.compute_distances(q, max_distance=20)
        input()
        pyb.stepSimulation(physicsClientId=gui_id)


    return

if __name__ == "__main__":
    main()
