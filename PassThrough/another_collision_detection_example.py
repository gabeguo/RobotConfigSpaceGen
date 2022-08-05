# Adapted from https://github.com/adamheins/pyb_utils

#!/usr/bin/env python
"""Example demonstrating collision detection and shortest distance queries."""
import numpy as np
import pybullet as pyb
import pybullet_data

from collision_detection import NamedCollisionObject, CollisionDetector
from itertools import combinations

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


def main():

    # main simulation server, with a GUI
    gui_id = pyb.connect(pyb.GUI)

    collision_bodies = load_environment(gui_id)

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
        gui_id,
        collision_bodies,
        collision_pairs
    )

    #pyb.resetDebugVisualizerCamera( cameraDistance=10, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0, 0, 0])

    MAX_ITERATIONS = 50
    for i in range(0, MAX_ITERATIONS + 1):
        # wait for user to press enter to continue
        input()

        # compute shortest distances for a configuration
        #q = [np.pi / 4, np.pi * 2 *  i / MAX_ITERATIONS, np.pi * 2 *  i / MAX_ITERATIONS]
        q = [np.pi / 4, np.pi * 3/4 * np.random.random(), np.pi * 3/4 * np.random.random()]
        d = col_detector.compute_distances(q, max_distance=20)
        in_col = col_detector.in_collision(q)

        if True:#in_col:
            q_deg = [x / np.pi * 180 for x in q]
            print(f"Configuration = {q_deg}")
            print(f"Distance to obstacles = {d}")
            print(f"In collision = {in_col}")


if __name__ == "__main__":
    main()
