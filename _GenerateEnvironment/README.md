# DeepCollide: Fully Reproducible Source Code

Supporting code for [DeepCollide: Scalable Data-Driven High DoF Configuration Space Modeling using Implicit Neural Representations](https://arxiv.org/abs/2305.15376)

Code has been validated end-to-end on two machines (listed in paper).

# Reproducing the Results

Run in the following order:

## To setup the Python environment:

Use Python 3.9.16. Make sure you have at least one GPU on your device.

    bash setup.sh

## To generate robot workspaces:

Section V-A: "Experimental Setup: Environments"

    bash run_environment_generation.sh

## To get Pareto frontiers:

Section VI: "Impact of Model"

    CUDA_VISIBLE_DEVICES=x bash run_models.sh
    bash run_pareto.sh

## To get DoF results:

Section VII: "Impact of DoF"

    CUDA_VISIBLE_DEVIECS=x bash run_dof_eval.sh

## To get collision results:

Section VIII: "Impact of Collision Density"

    CUDA_VISIBLE_DEVICES=x bash run_num_obstacles_eval.sh

## To get train size results:

Section IX: "Impact of Sample Size"

    CUDA_VISIBLE_DEVICES=x bash run_train_size_eval.sh

## To graph DoF, train size, and collision density:

Sections VI-IX (the previous commands generate the data, but do not make the super awesome plots)

    bash graph_experiments.sh
    
---

## Miscellaneous (Unnecessary)

We used a custom collision detection function [from Adam Heins](https://github.com/adamheins/pyb_utils), instead of the standard performCollisionDetection() with getContactPoints() in PyBullet. So, we have a script to measure their consistency.

    bash check_collision_detection_consistency.sh

Spoiler: *They are consistent.* The main difference is that PyBullet checks floor collisions, while we don't, since our focus is on obstacles (we can customize Adam Heins's method to detect specific objects). Also, PyBullet seems to return a few false positives: in a very small number of cases (~1%), it says there was a collision, but it says that the colliding objects are separated by some small distance (~1% of cube width), which would mean that there is no collision. 
