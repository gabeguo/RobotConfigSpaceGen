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
from constants import *

from PIL import Image

def load_environment(client_id, NUM_OBSTACLES, obstacle_positions, obstacle_orientations):
    assert len(obstacle_positions) == len(obstacle_orientations)

    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    # arm_id = [None for i in range(NUM_ROBOTS)]

    arm_id = pyb.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        baseOrientation=pyb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=1
    )

    obstacle_ids = [\
        pyb.loadURDF(
            "cube.urdf", obstacle_positions[i], \
            baseOrientation=pyb.getQuaternionFromEuler(obstacle_orientations[i]), \
            useFixedBase=True, globalScaling=0.5, physicsClientId=client_id
        ) \
        for i in range(len(obstacle_positions))
    ]

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot{}".format(i) : arm_id[i] for i in range(1)
    } + \
    {
        "obstacle{}".format(i): obstacle_ids[i] for i in range(NUM_OBSTACLES)
    }
    return bodies

def write_collision_data(fields, data):
    assert len(fields) == len(data[0])
    with open('collision_data.csv', 'w') as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(data)
    return

def main(NUM_ITERATIONS = 10000, NUM_OBSTACLES = 4, \
    obstacle_positions=[[0, -0.25, 0], [0, 0.25, 0], [-0.25, 0], [0.25, 0, 0]], \
    obstacle_orientations=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]):

    assert NUM_OBSTACLES == len(obstacle_positions) and len(obstacle_positions) == len(obstacle_orientations)

    # main simulation server
    sim_id = pyb.connect(pyb.DIRECT)

    collision_bodies = load_environment(sim_id, NUM_OBSTACLES, obstacle_positions, obstacle_orientations)

    for body in collision_bodies:
        #print(pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id))
        is_mesh = pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id)[0][2] == pyb.GEOM_MESH
        print(body, 'is mesh:', is_mesh)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking

    obstacles = [NamedCollisionObject("obstacle{}".format(i)) for i in range(NUM_OBSTACLES)]
    robotlinks = [NamedCollisionObject("robot0", "lbr_iiwa_link_{}".format(j)) for j in range(1, 7+1)] + \
        [NamedCollisionObject("robot0", None)]

    collision_objects = robotlinks + obstacles

    collision_pairs = [(the_link, the_obstacle) for the_link in robotlinks for the_obstacle in obstacles]

    #print(collision_pairs)

    col_detector = CollisionDetector(
        sim_id,
        collision_bodies,
        collision_pairs
    )

    COLLISION_DATA_LABELS = \
        ['robot{}_theta{}'.format(0, j) \
            for j in range(1, 7+1)] + \
        ['collision']

    _collision_data = []

    MAX_JOINT_ANGLE = [theta * np.pi / 180 for theta in [170, 120, 170, 120, 170, 120, 175]]
    # joint limits: https://www.researchgate.net/figure/Joint-limits-of-KUKA-LBR-iiwa-14-R820-45_tbl1_339394448

    # generate angles (since it is biased to generate them while also detecting if they collide)
    start = time.time()

    Q_trial = [] # Q_trial_robot[i] = configuration on trial i
    for i in range(0, NUM_ITERATIONS):
        Q_trial.append(list())
        for dof in range(7):
            Q_trial_robot[i].append(MAX_JOINT_ANGLE[dof] * 2 * np.random.random() - MAX_JOINT_ANGLE[dof])

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in generating', NUM_ITERATIONS, 'configurations:', elapsed, 'seconds')

    Q = [[theta for theta in trial] for trial in Q_trial_robot]

    # start detecting collisions
    start = time.time()

    for i in range(0, NUM_ITERATIONS):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances(Q_trial_robot[i], max_distance=0)
        in_col = (distances < 0).any()

        Q[i].append(int(in_col))
        _collision_data.append(Q[i])

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in checking', NUM_ITERATIONS, 'configurations for collision:', elapsed, 'seconds')
    # stop detecting collisions

    results = {TIME_COST : elapsed, SAMPLE_SIZE : NUM_ITERATIONS}

    write_collision_data(COLLISION_DATA_LABELS, _collision_data)

    ## GUI dummy demo of simulation starting point; does not move

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

    # https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd

    viewMatrix = pyb.computeViewMatrix(
        cameraEyePosition=[2, 2, 2],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[-0.5, -0.5, -0.25])
    projectionMatrix = pyb.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=4.1)
    width, height, rgbImg, depthImg, segImg = pyb.getCameraImage(
        width=1024,
        height=1024,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)

    print(type(rgbImg))
    im = Image.fromarray(rgbImg)
    im.save("graphs/{}_obstacles.png".format(len(obstacles)))

    ## cleanup

    # pyb.disconnect(physicsClientId=gui_id)
    pyb.disconnect(physicsClientId=sim_id)

    return results

if __name__ == "__main__":
    main()
