num_training_samples=30000
results_folder='ablation_results'
lr=1e-3
batch_size=512
train_percent=0.95
epochs=50
use_cuda="--use_cuda"
# use forward kinematics kernel!

for seed in 0 1 2
do
    dataset_name="3robots_25obstacles_seed${seed}_"
    echo "seed ${seed}"
    # DL
    for b in 1 2 5
    do
        for freq in 4 8 12
        do
            for sigma in 0.5 1 2
            do
                echo "DL ${use_cuda}: b ${b}, freq ${freq}, sigma ${sigma}"
                echo "vanilla"
                python compare_models.py --model_name "DL" --forward_kinematics_kernel \
                    --num_training_samples $num_training_samples \
                    --dataset_name $dataset_name \
                    --bias $b --num_freq $freq --sigma $sigma \
                    --lr $lr --batch_size $batch_size --train_percent $train_percent --epochs $epochs \
                    --results_folder $results_folder \
                    $use_cuda
                echo "disable_skip"
                python compare_models.py --model_name "DL" --forward_kinematics_kernel \
                    --num_training_samples $num_training_samples \
                    --dataset_name $dataset_name \
                    --bias $b --num_freq $freq --sigma $sigma \
                    --lr $lr --batch_size $batch_size --train_percent $train_percent --epochs $epochs \
                    --results_folder $results_folder \
                    $use_cuda --disable_skip_connection
                echo "disable_batchnorm"
                python compare_models.py --model_name "DL" --forward_kinematics_kernel \
                    --num_training_samples $num_training_samples \
                    --dataset_name $dataset_name \
                    --bias $b --num_freq $freq --sigma $sigma \
                    --lr $lr --batch_size $batch_size --train_percent $train_percent --epochs $epochs \
                    --results_folder $results_folder \
                    $use_cuda --disable_batchnorm
            done
        done
        echo "DL ${use_cuda}: b ${b}, disable positional encoding"
        python compare_models.py --model_name "DL" --forward_kinematics_kernel \
            --num_training_samples $num_training_samples \
            --dataset_name $dataset_name \
            --bias $b --num_freq 0 --sigma 0 \
            --lr $lr --batch_size $batch_size --train_percent $train_percent --epochs $epochs \
            --results_folder $results_folder \
            $use_cuda
    done
done