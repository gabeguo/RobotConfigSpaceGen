# DeepCollide: Fully Reproducible Source Code

Supporting code for [DeepCollide: Scalable Data-Driven High DoF Configuration Space Modeling using Implicit Neural Representations](https://arxiv.org/abs/2305.15376)

Code has been validated end-to-end on two machines (listed in paper).

## To setup the Python environment:

Use Python 3.9.16. Make sure you have at least one GPU on your device.

  bash setup.sh

## To generate robot workspaces:

  bash run_environment_generation.sh

## To get Pareto frontiers:

  CUDA_VISIBLE_DEVICES=x bash run_models.sh
  bash run_pareto.sh

## To get DoF results:

  CUDA_VISIBLE_DEVIECS=x bash run_dof_eval.sh

## To get collision results:

  CUDA_VISIBLE_DEVICES=x bash run_num_obstacles_eval.sh

## To get train size results:

  CUDA_VISIBLE_DEVICES=x bash run_train_size_eval.sh

## To graph DoF, train size, and collision density:

  bash graph_experiments.sh
