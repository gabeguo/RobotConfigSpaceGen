# Adapted from https://github.com/adamheins/pyb_utils

#!/usr/bin/env python
"""Example demonstrating collision detection and shortest distance queries."""
import numpy as np
import pybullet as pyb
import pybullet_data

from collision_detection import NamedCollisionObject, CollisionDetector


def load_environment(client_id):
    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id
    )

    # KUKA iiwa robot arm
    kuka_id = pyb.loadURDF(
        "kuka_iiwa/model.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=2
    )

    # some cubes for obstacles
    cube1_id = pyb.loadURDF(
        "cube.urdf", [1, 1, 0.5], useFixedBase=True, physicsClientId=client_id
    )
    cube2_id = pyb.loadURDF(
        "cube.urdf", [-1, -1, 0.5], useFixedBase=True, physicsClientId=client_id
    )
    cube3_id = pyb.loadURDF(
        "cube.urdf", [1, -1, 0.5], useFixedBase=True, physicsClientId=client_id
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot": kuka_id,
        "ground": ground_id,
        "cube1": cube1_id,
        "cube2": cube2_id,
        "cube3": cube3_id,
    }
    return bodies


def main():

    # main simulation server, with a GUI
    gui_id = pyb.connect(pyb.GUI)

    # simulation server only used for collision detection
    #col_id = pyb.connect(pyb.DIRECT)

    # add bodies to both of the environments
    #bodies = load_environment(gui_id)
    #collision_bodies = load_environment(col_id)
    collision_bodies = load_environment(gui_id)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking
    ground = NamedCollisionObject("ground")
    cube1 = NamedCollisionObject("cube1")
    cube2 = NamedCollisionObject("cube2")
    cube3 = NamedCollisionObject("cube3")
    link7 = NamedCollisionObject("robot", "lbr_iiwa_link_7")  # last link

    col_detector = CollisionDetector(
        gui_id,#col_id,
        collision_bodies,
        [(link7, ground), (link7, cube1), (link7, cube2), (link7, cube3)],
    )

    pyb.resetDebugVisualizerCamera( cameraDistance=10, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

    MAX_ITERATIONS = 50
    for i in range(0, MAX_ITERATIONS + 1):
        # wait for user to press enter to continue
        input()
        # compute shortest distances for a random configuration
        q = np.pi * (np.random.random(7) - 0.5)#q = [np.pi * 2 *  i / MAX_ITERATIONS for j in range(7)]#
        d = col_detector.compute_distances(q)
        in_col = col_detector.in_collision(q)

        print(f"Configuration = {q}")
        print(f"Distance to obstacles = {d}")
        print(f"In collision = {in_col}")

        # the main GUI-based simulation is not affected (i.e., the robot
        # doesn't move)
        #pyb.stepSimulation(physicsClientId=gui_id)


if __name__ == "__main__":
    main()
