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

import os

from PIL import Image

def load_environment(client_id, NUM_OBSTACLES, obstacle_positions, obstacle_orientations, obstacle_scale):
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

    cubeShape = pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
    sphereShape = pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=0.05)
    cylinderShape = pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=0.05, height=0.1)
    possibleShapes = [cubeShape, sphereShape, cylinderShape]
    obstacle_ids = [pyb.createMultiBody(baseMass=1, baseCollisionShapeIndex=possibleShapes[i%3], \
                                        basePosition=obstacle_positions[i], \
                                        baseOrientation=pyb.getQuaternionFromEuler(obstacle_orientations[i])) \
                                    for i in range(len(obstacle_positions))]
    planeId = pyb.loadURDF('plane.urdf')

    # add robot
    bodies = {
        "robot{}".format(0) : arm_id,
        "plane" : planeId
    }
    # also add obstacles
    bodies.update({
        "obstacle{}".format(i): obstacle_ids[i] for i in range(NUM_OBSTACLES)
    })
    return bodies

def write_collision_data(fields, data):
    assert len(fields) == len(data[0])
    with open('collision_data.csv', 'w') as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(data)
    return

def main(NUM_ITERATIONS=10000, NUM_OBSTACLES=50, obstacle_scale=0.1, SEED=0):
    np.random.seed(SEED)

    obstacle_positions = np.random.rand(NUM_OBSTACLES, 3)
    obstacle_positions[:, 0] -= 0.5
    obstacle_positions[:, 1] -= 0.5
    obstacle_positions[:, 0:2] *= 1.25
    obstacle_orientations = 2 * np.pi * np.random.rand(NUM_OBSTACLES, 3)

    assert NUM_OBSTACLES == len(obstacle_positions) and len(obstacle_positions) == len(obstacle_orientations)

    # main simulation server
    sim_id = pyb.connect(pyb.DIRECT)

    collision_bodies = load_environment(sim_id, NUM_OBSTACLES, obstacle_positions, obstacle_orientations, obstacle_scale)

    for body in collision_bodies:
        #print(pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id))
        is_mesh = pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id)[0][2] == pyb.GEOM_MESH
        # print(body, 'is mesh:', is_mesh)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking

    obstacles = [NamedCollisionObject("obstacle{}".format(i)) for i in range(NUM_OBSTACLES)]
    robotlinks = [NamedCollisionObject("robot0", "lbr_iiwa_link_{}".format(j)) for j in range(1, 7+1)] + \
        [NamedCollisionObject("robot0", None)]

    collision_objects = robotlinks + obstacles + ["plane"]

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
            Q_trial[i].append(MAX_JOINT_ANGLE[dof] * 2 * np.random.random() - MAX_JOINT_ANGLE[dof])

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in generating', NUM_ITERATIONS, 'configurations:', elapsed, 'seconds')

    Q = [[theta for theta in trial] for trial in Q_trial]

    # start detecting collisions
    start = time.time()

    for i in range(0, NUM_ITERATIONS):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances(Q_trial[i], max_distance=0)
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

    cameraEyePositions = [[3.5, 0, 1.5], [0, 3.5, 1.5], [-3.5, 0, 1.5], [0, -3.5, 1.5]]
    for i in range(len(cameraEyePositions)):
        cameraEyePosition = cameraEyePositions[i]
        viewMatrix = pyb.computeViewMatrix(
            cameraEyePosition=cameraEyePosition,
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1])
        projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=15)
        width, height, rgbImg, depthImg, segImg = pyb.getCameraImage(
            width=1024,
            height=1024,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        os.makedirs('graphs', exist_ok=True)

        print(type(rgbImg))
        im = Image.fromarray(rgbImg)
        im.save("graphs/{}_obstacles_trial{}.png".format(len(obstacles), i))

    ## cleanup

    # pyb.disconnect(physicsClientId=gui_id)
    pyb.disconnect(physicsClientId=sim_id)

    return results

if __name__ == "__main__":
    main()
