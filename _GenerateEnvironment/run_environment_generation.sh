#!/bin/bash

## 1) Increasing DoF experiment: impact of # DoF on Fastron performance
for num_robots in {1..4}
do
    for seed in {0..2}
    do
        python main.py --num_robots $num_robots --seed $seed \
        --num_obstacles 25 \
        --num_samples 50000 \
    done
done

## Is decreasing performance in high DoF due to changing collision density?
## 2) Hold DoF constant, adjust collision density
for num_obstacles in {10..60..10}
do
    for seed in {0..2}
    do
        # 2a) Is this due to robot-robot collision?
        python main.py --num_obstacles $num_obstacles --seed $seed \
        --num_robots 3 \
        --num_samples 50000 \
        --min_robot_robot_distance 0.5 \
        --max_robot_robot_distance 2.0 \
        --keyword_name 'collidingRobots'
        # 2b) Or robot-obstacle collision?
        python main.py --num_obstacles $num_obstacles --seed $seed \
        --num_robots 3 \
        --num_samples 50000 \
        --min_robot_robot_distance 2.5 \
        --max_robot_robot_distance 5.0 \
        --keyword_name 'separateRobots'
    done
done

## Or is decreasing performance in high DoF due to undersampling (curse of dimensionality)?
## 3) Hold DoF and collision density constant, adjust number of samples (can just sample a lot, and then take subset later)
for seed in {0..2}
do
    python main.py --num_samples 1000000 --seed $seed \
    --num_obstacles 25 \
    --num_robots 3 \
    --keyword_name '${num_samples}Samples'
done

## Possible arguments
# --num_robots 4 \
# --num_obstacles 25 \
# --num_samples 50000 \
# --obstacle_scale 0.1 \
# --min_robot_robot_distance 0.5 \
# --max_robot_robot_distance 5.0 \
# --min_robot_obstacle_distance 0.5 \
# --max_robot_obstacle_distance 1.25 \
# --min_obstacle_obstacle_distance 0.1 \
# --max_obstacle_obstacle_distance 10.0 \
# --min_robot_x -1.5 \
# --max_robot_x 1.5 \
# --min_robot_y -1.5 \
# --max_robot_y 1.5 \
# --min_obstacle_x -2.0 \
# --max_obstacle_x 2.0 \
# --min_obstacle_y -2.0 \
# --max_obstacle_y 2.0 \
# --min_obstacle_z 0.1 \
# --max_obstacle_z 1.75 \
# --seed 0