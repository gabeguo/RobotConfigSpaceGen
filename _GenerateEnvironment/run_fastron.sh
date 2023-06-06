# python fastron_benchmark.py \
#     --num_training_samples 25000 \
#     --dataset_name "3robots_25obstacles_seed0_" \
#     --g 5 \
#     --beta 500 \
#     --maxUpdates 20000 \
#     --maxSupportPoints 25000 \
#     --forward_kinematics_kernel

#!/bin/bash

## 1) Increasing DoF experiment: impact of # DoF on Fastron performance
echo "experiment 1: increase DoF"
for num_robots in {1..4}
do
    echo ""
    echo "${num_robots} robots"
    for seed in {0..2}
    do
        echo "seed ${seed}"
        dataset_name="${num_robots}robots_25obstacles_seed${seed}_"
        # try four Fastron settings per workspace
        for beta in 1 500
        do
            for use_kernel in "" "--forward_kinematics_kernel"
            do
                echo "beta ${beta} use_kernel ${use_kernel}"
                python fastron_benchmark.py $use_kernel \
                    --num_training_samples 25000 \
                    --dataset_name $dataset_name \
                    --g 5 \
                    --beta $beta \
                    --maxUpdates 25000 \
                    --maxSupportPoints 25000
                # max updates is much larger than paper (5000)
            done
        done
    done
done

# ## Is decreasing performance in high DoF due to changing collision density?
# ## 2) Hold DoF constant, adjust collision density
# echo "experiment 2: adjust collision density"
# for num_obstacles in {10..60..10}
# do
#     for seed in {0..2}
#     do
#         for robot_collision in 'collidingRobots' 'separateRobots'
#         do
#             dataset_name="3robots_${num_obstacles}obstacles_seed${seed}_${robot_collision}"
#             # try four Fastron settings per workspace
#             for beta in 1 500
#             do
#                 for use_kernel in "" "--forward_kinematics_kernel"
#                 do
#                     python fastron_benchmark.py $use_kernel \
#                         --num_training_samples 25000 \
#                         --dataset_name $dataset_name \
#                         --g 5 \
#                         --beta $beta \
#                         --maxUpdates 25000 \
#                         --maxSupportPoints 25000
#                 done
#             done
#         done
#     done
# done

# ## Or is decreasing performance in high DoF due to undersampling (curse of dimensionality)?
# ## 3) Hold DoF and collision density constant, adjust number of samples (can just sample a lot, and then take subset later)
# echo "experiment 3: adjust number of samples"
# for seed in {0..2}
# do
#     dataset_name="3robots_25obstacles_seed${seed}_1000000Samples"
#     for num_samples in 1000 10000 100000 500000 900000
#     do
#         # try four Fastron settings per workspace
#         for beta in 1 500
#         do
#             for use_kernel in "" "--forward_kinematics_kernel"
#             do
#                 python fastron_benchmark.py $use_kernel \
#                     --num_training_samples $num_samples \
#                     --dataset_name $dataset_name \
#                     --g 5 \
#                     --beta $beta \
#                     --maxUpdates 1000000 \
#                     --maxSupportPoints $num_samples
#             done
#         done
#     done
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
