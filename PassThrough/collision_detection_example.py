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
        "basic_arm.urdf",#"kuka_iiwa/model.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
        globalScaling=1
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
    col_id = pyb.connect(pyb.DIRECT)

    # add bodies to both of the environments
    bodies = load_environment(gui_id)
    collision_bodies = load_environment(col_id)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking
    ground = NamedCollisionObject("ground")
    cube1 = NamedCollisionObject("cube1")
    cube2 = NamedCollisionObject("cube2")
    cube3 = NamedCollisionObject("cube3")
    last_link = NamedCollisionObject("robot", "top_arm")#link7 = NamedCollisionObject("robot", "lbr_iiwa_link_7")  # last link
    mid_link = NamedCollisionObject("robot", "middle_arm")
    first_link = NamedCollisionObject("robot", "bottom_arm")

    col_detector = CollisionDetector(
        col_id,
        collision_bodies,
        [(last_link, ground), (last_link, cube1), (last_link, cube2), (last_link, cube3), \
        (mid_link, ground), (mid_link, cube1), (mid_link, cube2), (mid_link, cube3), \
        (first_link, ground), (first_link, cube1), (first_link, cube2), (first_link, cube3)]
    )

    while True:

        # COMPUTATION FOR SIMULATION

        # compute shortest distances for a random configuration
        q = 2 * np.pi * (np.random.random(3) - 0.5)#q = np.pi * (np.random.random(7) - 0.5)
        d = col_detector.compute_distances(q)
        in_col = col_detector.in_collision(q)

        # COMPUTATION FOR PHYSICAL

        # START GABE MODIFICATION
        robot_id = bodies['robot']
        # put the robot in the given configuration
        for i in range(pyb.getNumJoints(robot_id, physicsClientId=gui_id)):
            pyb.resetJointState(robot_id, i, q[i], physicsClientId=gui_id)
        # END GABE MODIFICATION

        # USER INPUT

        # wait for user to press enter to continue
        #input()

        # UPDATE DISPLAY

        # the main GUI-based simulation is not affected (i.e., the robot
        # doesn't move)
        pyb.stepSimulation(physicsClientId=gui_id)

        # PRINT INFO

        print('\n***\nNew position:')
        print(f"Configuration (radians) = {q}")
        print(f"Configuration (degrees) = {180 / np.pi * q}")
        print(f"Distance to obstacles = {d}")
        print(f"In collision = {in_col}")

        input()
        print('side view')
        pyb.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0, 0, 0], physicsClientId=gui_id)
        input()
        print('other side view')
        pyb.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 0], physicsClientId=gui_id)
        input()
        print('top view')
        pyb.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0], physicsClientId=gui_id)
        input()

if __name__ == "__main__":
    main()
