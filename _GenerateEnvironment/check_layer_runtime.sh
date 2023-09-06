for seed in 0 1 2
do
    for num_robots in 1 2 3 4 5 6
    do
        python compare_models.py --model_name 'DL' --forward_kinematics_kernel \
            --num_training_samples 30000 --num_testing_samples 5000 \
            --dataset_name "${num_robots}robots_25obstacles_seed${seed}_" \
            --bias 1 --num_freq 12 --sigma 1 \
            --lr 1e-3 --batch_size 512 --train_percent 0.95 --epochs 40 \
            --results_folder 'layer_runtime_tests' \
            --time_layers
    done
done