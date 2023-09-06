# for seed in 0 1 2
# do
#     for num_robots in 1 2 3 4 5 6
#     do
#         python compare_models.py --model_name 'DL' --forward_kinematics_kernel \
#             --num_training_samples 30000 --num_testing_samples 5000 \
#             --dataset_name "${num_robots}robots_25obstacles_seed${seed}_" \
#             --bias 1 --num_freq 12 --sigma 1 \
#             --lr 1e-3 --batch_size 512 --train_percent 0.95 --epochs 40 \
#             --results_folder 'layer_runtime_tests' \
#             --time_layers
#     done
# done

num_training_samples=30000
results_folder='layer_runtime_tests'
# use forward kinematics kernel!

for num_robots in 1 2 3 4 5 6
do
    echo "${num_robots} robots"
    for seed in 0 1 2
    do
        dataset_name="${num_robots}robots_25obstacles_seed${seed}_"

        # DL
        for freq in 4 8 12
        do
            for b in 1 2 5
            do
                for sigma in 0.5 1 2
                do
                    echo "DL: freq ${freq}, b ${b}, sigma ${sigma}"
                    python compare_models.py --model_name 'DL' --forward_kinematics_kernel \
                        --num_training_samples $num_training_samples \
                        --dataset_name $dataset_name \
                        --bias $b --num_freq $freq --sigma $sigma \
                        --lr 1e-3 --batch_size 512 --train_percent 0.95 --epochs 50 \
                        --results_folder $results_folder \
                        --time_layers
                done
            done
        done
    done
done