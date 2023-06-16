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

import json
import argparse

MAX_JOINT_ANGLE = [theta * np.pi / 180 for theta in [170, 120, 170, 120, 170, 120, 175]]
# joint limits: https://www.researchgate.net/figure/Joint-limits-of-KUKA-LBR-iiwa-14-R820-45_tbl1_339394448

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


def load_environment(client_id, num_obstacles, obstacle_positions, obstacle_orientations, obstacle_scale, \
                     num_robots, robot_positions, robot_orientations):
    assert len(obstacle_positions) == len(obstacle_orientations)

    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    arm_id = [None for i in range(num_robots)]

    for i in range(num_robots):
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
        "robot{}".format(i) : arm_id[i] for i in range(num_robots)
    }
    # add plane
    bodies["plane"] = planeId
    # also add obstacles
    bodies.update({
        "obstacle{}".format(i): obstacle_ids[i] for i in range(num_obstacles)
    })
    return bodies

def write_collision_data(fields, data, args):
    assert len(fields) == len(data[0])
    filename = f"{DATA_FOLDER}/collision_data_{args.num_robots}robots_{args.num_obstacles}obstacles_seed{args.seed}_{args.keyword_name}.csv"
    with open(filename, 'w') as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(data)
    return

def data_to_np(data, field_name, args):
    data = np.array(data)
    filename = f"{DATA_FOLDER}/{field_name}_{args.num_robots}robots_{args.num_obstacles}obstacles_seed{args.seed}_{args.keyword_name}.npy"
    np.save(filename, data)
    return

def configs_to_np(configs, args):
    data_to_np(configs, 'configs', args)
    return

def labels_to_np(labels, args):
    labels = np.array([1 if y > 0 else -1 for y in labels])
    data_to_np(labels, 'labels', args)
    return

def link_pos_to_np(all_link_pos, args):
    data_to_np(all_link_pos, 'linkPositions', args)
    return

# Thanks ChatGPT!
# Saves args and results as dict
def save_results(results, args):
    # convert args to dict
    args_dict = vars(args)
    args_dict.update(results)

    # construct the filename
    filename = f"{DATA_FOLDER}/argsAndResults_{args.num_robots}robots_{args.num_obstacles}obstacles_seed{args.seed}_{args.keyword_name}.json"
    
    # write the JSON file
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

    return

def take_pictures(args):
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

        os.makedirs('workspace_vis', exist_ok=True)

        print(type(rgbImg))
        im = Image.fromarray(rgbImg)
        im.save("workspace_vis/{}robots_{}obstacles_seed{}_sample{}{}.png".format(\
            args.num_robots, args.num_obstacles, args.seed, i, '_' + args.keyword_name if len(args.keyword_name) > 0 else '')\
        )
    return

# Thanks ChatGPT!
# Assume object is centered at 0, 0, 0
def getRobotBoundingBoxShapes(robot_id):
    bounding_box_shapes = list()
    for jointIndex in range(pyb.getNumJoints(robot_id)):
        # Get the collision shape data for the joint
        shapeData = pyb.getCollisionShapeData(robot_id, jointIndex)
        # Check if the shape is a mesh
        if shapeData[0][2][0] == pyb.GEOM_MESH:
            # The second item of the shape data contains the dimensions of the bounding box of the mesh
            dimensions = shapeData[0][2][1]
            print(f"Joint {jointIndex} is a mesh with bounding box dimensions {dimensions}")

            # approximate it as a box (bounding box)
            box_points = [(-dimensions[0]/2, -dimensions[1]/2, -dimensions[2]/2),
                        (-dimensions[0]/2, -dimensions[1]/2, dimensions[2]/2),
                        (-dimensions[0]/2, dimensions[1]/2, -dimensions[2]/2),
                        (-dimensions[0]/2, dimensions[1]/2, dimensions[2]/2),
                        (dimensions[0]/2, -dimensions[1]/2, -dimensions[2]/2),
                        (dimensions[0]/2, -dimensions[1]/2, dimensions[2]/2),
                        (dimensions[0]/2, dimensions[1]/2, -dimensions[2]/2),
                        (dimensions[0]/2, dimensions[1]/2, dimensions[2]/2)]
            bounding_box_shapes.append(box_points)
    return bounding_box_shapes

# Thanks ChatGPT!
# Assume object is centered at 0, 0, 0
def getObstacleBoundingBoxShape(boxId):
    box_shape_data = pyb.getCollisionShapeData(boxId, -1)
    box_dimensions = box_shape_data[0][2][1] # TODO: check that this is bounding box
    # Create a set of points for the box
    obstacle_points = [(-box_dimensions[0]/2, -box_dimensions[1]/2, -box_dimensions[2]/2),
                    (-box_dimensions[0]/2, -box_dimensions[1]/2, box_dimensions[2]/2),
                    (-box_dimensions[0]/2, box_dimensions[1]/2, -box_dimensions[2]/2),
                    (-box_dimensions[0]/2, box_dimensions[1]/2, box_dimensions[2]/2),
                    (box_dimensions[0]/2, -box_dimensions[1]/2, -box_dimensions[2]/2),
                    (box_dimensions[0]/2, -box_dimensions[1]/2, box_dimensions[2]/2),
                    (box_dimensions[0]/2, box_dimensions[1]/2, -box_dimensions[2]/2),
                    (box_dimensions[0]/2, box_dimensions[1]/2, box_dimensions[2]/2)]
    return obstacle_points

# Thanks ChatGPT!
"""
Returns: robotIds, robotBoundingBoxShapes, obstacleIds, obstacleBoundingBoxShapes
"""
def getBoundingBoxShapes(collision_bodies):
    robotBoundingBoxShapes = list() # list of list of lists (dim 0: robot number; dim 1: joint number; dim 2: individual bounding box corner)
    obstacleBoundingBoxShapes = list() # list of lists (dim 0: obstacle number; dim 1: individual bounding box corner)
    robotIds = list()
    obstacleIds = list()
    for body_name in sorted(collision_bodies):
        if 'robot' in body_name: # robot
            robot_id = collision_bodies[body_name]
            robotBoundingBoxShapes.append(getRobotBoundingBoxShapes(robot_id))
            robotIds.append(robot_id)
        else: # obstacle
            boxId = collision_bodies[body_name]
            obstacleBoundingBoxShapes.append(getObstacleBoundingBoxShape(boxId))
            obstacleIds.append(boxId)
    return robotIds, robotBoundingBoxShapes, obstacleIds, obstacleBoundingBoxShapes




from scipy.spatial.transform import Rotation as R

# Thanks ChatGPT!
# Function to apply pose transformation
def apply_transform(points, pose):
    pos, orn = pose
    rotation = R.from_quat([orn[0], orn[1], orn[2], orn[3]])

    # Apply rotation and translation to each point
    transformed_points = []
    for point in points:
        transformed_points.append(rotation.apply(point) + pos)

    return transformed_points

import sys
sys.path.append('openGJK/examples/cython')
import openGJK as opengjk

def main():
    args = get_args()

    np.random.seed(args.seed)

    robot_positions = generate_coordinates(n=args.num_robots, \
                        min_distance_fixed=0, max_distance_fixed=np.infty,
                        min_distance_gen=args.min_robot_robot_distance, max_distance_gen=args.max_robot_robot_distance,
                        fixed_points=None, 
                        x_range=(args.min_robot_x, args.max_robot_x), y_range=(args.min_robot_y, args.max_robot_y),
                        z_range=None, z_fixed=0)
    robot_orientations = 2 * np.pi * np.random.rand(args.num_robots, 3)
    robot_orientations[:,0:2] = 0

    obstacle_positions = generate_coordinates(n=args.num_obstacles, \
                        min_distance_fixed=args.min_robot_obstacle_distance, max_distance_fixed=args.max_robot_obstacle_distance, \
                        min_distance_gen=args.min_obstacle_obstacle_distance, max_distance_gen=args.max_obstacle_obstacle_distance, \
                        fixed_points=robot_positions, 
                        x_range=(args.min_obstacle_x, args.max_obstacle_x), 
                        y_range=(args.min_obstacle_y, args.max_obstacle_y), 
                        z_range=(args.min_obstacle_z, args.max_obstacle_z), z_fixed=None)
    obstacle_orientations = 2 * np.pi * np.random.rand(args.num_obstacles, 3)

    print(robot_positions)
    print(obstacle_positions)

    assert args.num_obstacles== len(obstacle_positions) and len(obstacle_positions) == len(obstacle_orientations)

    # main simulation server
    sim_id = pyb.connect(pyb.DIRECT)

    collision_bodies = load_environment(client_id=sim_id,
        num_obstacles=args.num_obstacles, obstacle_positions=obstacle_positions, obstacle_orientations=obstacle_orientations,
        obstacle_scale=args.obstacle_scale,
        num_robots=args.num_robots, 
        robot_positions=robot_positions, robot_orientations=robot_orientations)

    for body in collision_bodies:
        #print(pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id))
        is_mesh = pyb.getCollisionShapeData(collision_bodies[body], -1, sim_id)[0][2] == pyb.GEOM_MESH
        # print(body, 'is mesh:', is_mesh)

    # define bodies (and links) to use for shortest distance computations and
    # collision checking

    obstacles = [NamedCollisionObject("obstacle{}".format(i)) for i in range(args.num_obstacles)]

    robotlinks = [\
        [NamedCollisionObject("robot{}".format(i), "lbr_iiwa_link_{}".format(j)) \
                    for j in range(1, 7+1)] + \
        [NamedCollisionObject("robot{}".format(i), None)] \
                    for i in range(args.num_robots)]

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

    collision_data_labels = \
        ['robot{}_theta{}'.format(i, j) \
            for i in range(args.num_robots) \
            for j in range(1, 7+1)] + \
        ['collision']

    _collision_data = []

    # generate angles (since it is biased to generate them while also detecting if they collide)
    start = time.time()

    Q_trial_robot = [] # Q_trial_robot[i][j] = configuration of robot j on trial i
    normalized_configurations = list() # normalized_configurations[i][j] = Q_trial_robot[i][j], but normalized to [-1, +1]
    for i in range(0, args.num_samples):
        Q_trial_robot.append(list())
        normalized_configurations.append(list())
        for j in range(args.num_robots):
            Q_trial_robot[i].append(list())
            normalized_configurations[i].append(list())
            for dof in range(7):
                curr_normalized_config = 2 * np.random.random() - 1 # [-1, +1]
                Q_trial_robot[i][j].append(MAX_JOINT_ANGLE[dof] * curr_normalized_config)
                normalized_configurations[i][j].append(curr_normalized_config)

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in generating', args.num_samples, 'configurations:', elapsed, 'seconds')

    assert len(Q_trial_robot) == args.num_samples

    # this contains NORMALIZED CONFIGURATIONS!
    Q = [[theta for robot_data in trial for theta in robot_data] for trial in normalized_configurations]
    all_configs = np.array(Q)
    labels = list()

    all_link_pos = list()

    # start detecting collisions
    start = time.time()

    for i in tqdm(range(0, args.num_samples)):
        # compute shortest distances for a configuration
        distances = col_detector.compute_distances_multi_robot(Q_trial_robot[i], max_distance=0)
        in_col = (distances < 0).any()

        Q[i].append(int(in_col))
        _collision_data.append(Q[i])

        labels.append(1 if in_col else -1)

        # this computes forward kinematics!
        all_link_pos.append(list())
        for body_name in sorted(collision_bodies):
            if 'robot' not in body_name:
                continue
            robot_id = collision_bodies[body_name]
            for link_id in range(0, 7):
                link_pos = pyb.getLinkState(robot_id, link_id)[0]
                all_link_pos[i].extend(link_pos)

    end = time.time()
    elapsed = round(end - start, 3)
    print('time elapsed in checking', args.num_samples, 'configurations for collision:', elapsed, 'seconds')
    # stop detecting collisions








    # TODO: separate forward kinematics pass
    # TODO: separate GJK bounding box computation from collision detection

    # Get information about all the items
    robotIds, robot_bounding_boxes, obstacleIds, obstacle_bounding_boxes = getBoundingBoxShapes(collision_bodies)

    # GJK collision detection
    for i in tqdm(range(0, args.num_samples)):

        for robot_id, the_robot in zip(robotIds, robot_bounding_boxes):
            assert isinstance(the_robot, list)
            for jointIndex, the_link in enumerate(the_robot):
                assert isinstance(the_link, list)
                linkState = pyb.getLinkState(robot_id, jointIndex)
                linkPose = linkState[0], linkState[1] # TODO: verify
                link_points = apply_transform(the_link, linkPose)

                for obstacle_id, the_obstacle in zip(obstacleIds, obstacle_bounding_boxes):
                    obstacleState = pyb.getBasePositionAndOrientation(obstacle_id)
                    obstaclePose = obstacleState[0], obstacleState[1]

                    # Apply the pose transformation to the points
                    obstacle_points = apply_transform(obstacle_points, obstaclePose)

                    # Run the GJK algorithm
                    distance = opengjk.pygjk(link_points, obstacle_points) # TODO: verify that these are np arrays

                    if distance > 0:
                        print(f"Collision detected between joint {jointIndex} and box {boxId}")







    results = {TIME_COST : elapsed, SAMPLE_SIZE : args.num_samples}

    os.makedirs(DATA_FOLDER, exist_ok=True)
    configs_to_np(all_configs, args)
    labels_to_np(labels, args)
    link_pos_to_np(all_link_pos, args)
    save_results(results, args)
    write_collision_data(collision_data_labels, _collision_data, args)

    ## GUI dummy demo of simulation starting point; does not move
    ## Take pictures
    take_pictures(args)

    ## cleanup

    # pyb.disconnect(physicsClientId=gui_id)
    pyb.disconnect(physicsClientId=sim_id)

    return

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_obstacles', type=int, default=25)
    parser.add_argument('--num_robots', type=int, default=3)
    parser.add_argument('--obstacle_scale', type=float, default=0.1)
    parser.add_argument('--min_robot_robot_distance', type=float, default=0.5)
    parser.add_argument('--max_robot_robot_distance', type=float, default=5.0)
    parser.add_argument('--min_robot_obstacle_distance', type=float, default=0.5)
    parser.add_argument('--max_robot_obstacle_distance', type=float, default=1.25)
    parser.add_argument('--min_obstacle_obstacle_distance', type=float, default=0.1)
    parser.add_argument('--max_obstacle_obstacle_distance', type=float, default=10.0)
    parser.add_argument('--min_robot_x', type=float, default=-1.5)
    parser.add_argument('--max_robot_x', type=float, default=1.5)
    parser.add_argument('--min_robot_y', type=float, default=-1.5)
    parser.add_argument('--max_robot_y', type=float, default=1.5)
    parser.add_argument('--min_obstacle_x', type=float, default=-2.0)
    parser.add_argument('--max_obstacle_x', type=float, default=2.0)
    parser.add_argument('--min_obstacle_y', type=float, default=-2.0)
    parser.add_argument('--max_obstacle_y', type=float, default=2.0)
    parser.add_argument('--min_obstacle_z', type=float, default=0.1)
    parser.add_argument('--max_obstacle_z', type=float, default=1.75)
    parser.add_argument('--keyword_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()