results_folder='surfaceSampling_experiment_results'
# use forward kinematics kernel!

# Indices 0-49999 are surface sampling, 50000-99999 are uniform sampling
train_lower=(25000 50000 37500)
train_upper=(50000 75000 62500)
test_lower=(0 95000)
test_upper=(5000 100000)

for train_group in 0 1 2
do
    for test_group in 0 1
    do
        echo "Train: ${train_lower[$train_group]} ${train_upper[$train_group]}"
        echo "Test: ${test_lower[$test_group]} ${test_upper[$test_group]}"
        for seed in 0 1 2
        do
            dataset_name="surface_sampling"

            # Fastron
            updates=50000
            support=50000
            for g in 1 2 5 10
            do
                for b in 1 10 500 1000
                do
                    echo "Fastron: support ${support}, updates ${updates}, g ${g}, b ${b}"
                    python compare_models.py --model_name 'Fastron' --forward_kinematics_kernel \
                        --train_indices $train_lower[$train_group] $train_upper[$train_group] \
                        --test_indices $test_lower[$test_group] $test_upper[$test_group] \
                        --dataset_name $dataset_name \
                        --g $g --beta $b --maxUpdates $updates --maxSupportPoints $support \
                        --results_folder $results_folder
                done
            done

            # DL
            for freq in 4 8 12
            do
                for b in 1 2 5
                do
                    for sigma in 0.5 1 2
                    do
                        for gpu_flag in '--use_cuda'
                        do
                            echo "DL ${gpu_flag}: freq ${freq}, b ${b}, sigma ${sigma}"
                            python compare_models.py --model_name "DL${gpu_flag}" --forward_kinematics_kernel \
                                --train_indices $train_lower[$train_group] $train_upper[$train_group] \
                                --test_indices $test_lower[$test_group] $test_upper[$test_group] \
                                --dataset_name $dataset_name \
                                --bias $b --num_freq $freq --sigma $sigma \
                                --lr 1e-3 --batch_size 512 --train_percent 0.95 --epochs 50 \
                                --results_folder $results_folder \
                                $gpu_flag
                        done
                    done
                done
            done
        done
    done
done