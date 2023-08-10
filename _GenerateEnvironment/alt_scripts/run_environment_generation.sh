# DIFF: Have some dummy experiment to see how fast GJK is, and also to verify low data case

# #!/bin/bash

# 0) See how much time FK takes
echo "dummy: see FK calculation time and GJK time"
#for num_robots in {1..6}
for num_robots in 4
do
    for seed in {7..9}
    do
        python generate_environment.py --num_robots $num_robots --seed $seed \
        --num_obstacles 26 \
        --num_samples 10000 \
        --keyword_name 'otherDummy'
    done
done

# ## 1) Increasing DoF experiment: impact of # DoF on Fastron performance
# echo "experiment 1: increase DoF"
# #for num_robots in {1..6}
# for num_robots in 5 6
# do
#     for seed in {0..2}
#     do
#         python generate_environment.py --num_robots $num_robots --seed $seed \
#         --num_obstacles 25 \
#         --num_samples 50000
#     done
# done

# ## Is decreasing performance in high DoF due to changing collision density?
# ## 2) Hold DoF constant, adjust collision density
# echo "experiment 2: adjust collision density"
# for num_obstacles in {10..60..10}
# do
#     for seed in {0..2}
#     do
#         # 2a) Is this due to robot-robot collision?
#         python generate_environment.py --num_obstacles $num_obstacles --seed $seed \
#         --num_robots 3 \
#         --num_samples 50000 \
#         --min_robot_robot_distance 0.5 \
#         --max_robot_robot_distance 2.0 \
#         --keyword_name 'collidingRobots'
#         # 2b) Or robot-obstacle collision?
#         python generate_environment.py --num_obstacles $num_obstacles --seed $seed \
#         --num_robots 3 \
#         --num_samples 50000 \
#         --min_robot_robot_distance 2.0 \
#         --max_robot_robot_distance 5.0 \
#         --keyword_name 'separateRobots'
#     done
# done

# ## Or is decreasing performance in high DoF due to undersampling: curse of dimensionality?
# ## 3) Hold DoF and collision density constant, adjust number of samples: can just sample a lot, and then take subset later
# echo "experiment 3: adjust number of samples"
# for seed in {0..2}
# do
#     python generate_environment.py --num_samples 1000000 --seed $seed \
#     --num_obstacles 25 \
#     --num_robots 3 \
#     --keyword_name '1000000Samples'
# done


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
