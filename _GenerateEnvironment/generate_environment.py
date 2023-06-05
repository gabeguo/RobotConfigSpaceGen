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
from tqdm import tqdm

# Thanks ChatGPT
def sample_points_inside_box(centers, length, width, height, num_points):
    center_indices = np.random.choice(range(len(centers)), size=len(centers), replace=True)
    x = centers[center_indices][0] + (np.random.rand(num_points) - 0.5) * length
    y = centers[center_indices][1] + (np.random.rand(num_points) - 0.5) * width
    z = centers[center_indices][2] + (np.random.rand(num_points) - 0.5) * height
    
    return np.vstack((x, y, z)).T

# Thanks ChatGPT
def sample_points_inside_sphere(center, radius, num_points):
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)
    w = np.random.rand(num_points)
    r = radius * np.cbrt(u)

    x = center[0] + r * np.sin(v * 2 * np.pi) * np.cos(w * 2 * np.pi)
    y = center[1] + r * np.sin(v * 2 * np.pi) * np.sin(w * 2 * np.pi)
    z = center[2] + r * np.cos(v * 2 * np.pi)
    
    return np.vstack((x, y, z)).T

# Thanks ChatGPT
def sample_points_inside_cylinder(center, height, radius, num_points):
    # Sample cylindrical coordinates.
    z = center[2] + (np.random.rand(num_points) - 0.5) * height
    theta = np.random.rand(num_points) * 2 * np.pi
    r = radius * np.sqrt(np.random.rand(num_points))  # Use sqrt to ensure uniform distribution.

    # Convert to Cartesian coordinates.
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.vstack((x, y, z)).T

def generate_coordinates(n, min_distance_fixed, max_distance_fixed, min_distance_gen, max_distance_gen,
                         fixed_points=None, x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), z_fixed=None):
    if isinstance(fixed_points, np.ndarray):
        fixed_points = fixed_points.tolist()
    points = [] if fixed_points is None else fixed_points.copy()
    generated_points = []
    while len(generated_points) < n:
        # Generate a random point
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = z_fixed if z_fixed is not None else np.random.uniform(*z_range)
        point = np.array([x, y, z])

        # Compute distances to fixed points
        if points:
            distances_fixed = np.linalg.norm(np.array(points) - point, axis=1)
        else:
            distances_fixed = np.array([min_distance_fixed, max_distance_fixed])

        # Compute distances to already generated points
        if generated_points:
            distances_gen = np.linalg.norm(np.array(generated_points) - point, axis=1)
        else:
            distances_gen = np.array([min_distance_gen, max_distance_gen])

        # If it's far enough from all existing points and generated points, and not too far from any fixed points or generated points, add it
        if (np.all(distances_fixed >= min_distance_fixed) 
                and np.any(distances_fixed <= max_distance_fixed) # just needds to be close to one other
                and np.all(distances_gen >= min_distance_gen)
                and np.any(distances_gen <= max_distance_gen)): # just needs to be close to one other
            generated_points.append(point)

    return np.array(generated_points)  # Return only the new points


def load_environment(client_id, NUM_OBSTACLES, obstacle_positions, obstacle_orientations, obstacle_scale, \
                     NUM_ROBOTS, robot_positions, robot_orientations):
    assert len(obstacle_positions) == len(obstacle_orientations)

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
            globalScaling=1
        )

    cubeShape = pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[obstacle_scale for i in range(3)])
    sphereShape = pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=obstacle_scale)
    #cylinderShape = pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=obstacle_scale, height=0.1)
    possibleShapes = [cubeShape, sphereShape]#, cylinderShape]
    obstacle_ids = [pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=possibleShapes[i%len(possibleShapes)], \
                                        basePosition=obstacle_positions[i], \
                                        baseOrientation=pyb.getQuaternionFromEuler(obstacle_orientations[i])) \
                                    for i in range(len(obstacle_positions))]
    planeId = pyb.loadURDF('plane.urdf')

    # add robots
    bodies = {
        "robot{}".format(i) : arm_id[i] for i in range(NUM_ROBOTS)
    }
    # add plane
    bodies["plane"] = planeId
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

def configs_to_np(configs):
    configs = np.array(configs)
    np.save('configs.npy', configs)
    return

def labels_to_np(labels):
    labels = np.array([1 if y > 0 else -1 for y in labels])
    np.save('labels.npy', labels)
    return

def main(NUM_ITERATIONS=100000, NUM_OBSTACLES=35, NUM_ROBOTS=3, obstacle_scale=0.1, SEED=0):
    np.random.seed(SEED)

    robot_positions = generate_coordinates(n=NUM_ROBOTS, \
                        min_distance_fixed=0, max_distance_fixed=10, \
                        min_distance_gen=0.5, max_distance_gen=5, \
                        fixed_points=None, 
                        x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), z_range=None, z_fixed=0)
    robot_orientations = 2 * np.pi * np.random.rand(NUM_ROBOTS, 3)
    robot_orientations[:,0:2] = 0

    obstacle_positions = generate_coordinates(n=NUM_OBSTACLES, \
                        min_distance_fixed=0.5, max_distance_fixed=1.25, \
                        min_distance_gen=0.1, max_distance_gen=10, \
                        fixed_points=robot_positions, 
                        x_range=(-2, 2), y_range=(-2, 2), z_range=(0.1, 1.75), z_fixed=None)
    obstacle_orientations = 2 * np.pi * np.random.rand(NUM_OBSTACLES, 3)

    print(robot_positions)
    print(obstacle_positions)

    assert NUM_OBSTACLES == len(obstacle_positions) and len(obstacle_positions) == len(obstacle_orientations)

    # main simulation server
    sim_id = pyb.connect(pyb.DIRECT)
    # Draw the x-axis
    pyb.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0])  # Red line
    # Draw the y-axis
    pyb.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0])  # Green line
    # Draw the z-axis
    pyb.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1])  # Blue line

    collision_bodies = load_environment(sim_id, \
                                        NUM_OBSTACLES, obstacle_positions, obstacle_orientations, obstacle_scale,\
                                        NUM_ROBOTS, robot_positions, robot_orientations)

    for body in collision_bodies:
        #print(pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id))
        is_mesh = pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id)[0][2] == pyb.GEOM_MESH
        # print(body, 'is mesh:', is_mesh)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking

    obstacles = [NamedCollisionObject("obstacle{}".format(i)) for i in range(NUM_OBSTACLES)]

    robotlinks = [\
        [NamedCollisionObject("robot{}".format(i), "lbr_iiwa_link_{}".format(j)) \
                    for j in range(1, 7+1)] + \
        [NamedCollisionObject("robot{}".format(i), None)] \
                    for i in range(NUM_ROBOTS)]

    collision_objects = [the_link for the_robotlinks in robotlinks for the_link in the_robotlinks] \
        + obstacles + ['plane']

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

    assert len(Q_trial_robot) == NUM_ITERATIONS

    Q = [[theta for robot_data in trial for theta in robot_data] for trial in Q_trial_robot]
    all_configs = np.array(Q)
    labels = list()

    # start detecting collisions
    start = time.time()

    for i in tqdm(range(0, NUM_ITERATIONS)):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances_multi_robot(Q_trial_robot[i], max_distance=0)
        in_col = (distances < 0).any()

        Q[i].append(int(in_col))
        _collision_data.append(Q[i])

        labels.append(1 if in_col else -1)

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in checking', NUM_ITERATIONS, 'configurations for collision:', elapsed, 'seconds')
    # stop detecting collisions

    results = {TIME_COST : elapsed, SAMPLE_SIZE : NUM_ITERATIONS}

    configs_to_np(all_configs)
    labels_to_np(labels)

    write_collision_data(COLLISION_DATA_LABELS, _collision_data)

    ## GUI dummy demo of simulation starting point; does not move

    cameraEyePositions = [[4, 0, 1.5], [0, 4, 1.5], [-4, 0, 1.5], [0, -4, 1.5]]
    for i in range(len(cameraEyePositions)):
        cameraEyePosition = cameraEyePositions[i]
        viewMatrix = pyb.computeViewMatrix(
            cameraEyePosition=cameraEyePosition,
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1])
        projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=10)
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

    # TODO: log time to get forward kinematics!!