results_folder='trainSize_experiment_results'
# use forward kinematics kernel!

for num_training_samples in 100 1000 5000 10000 20000 40000 60000 80000 100000
do
    echo "${num_training_samples} samples"
    for seed in 0 1 2
    do
        dataset_name="3robots_25obstacles_seed${seed}_1000000Samples"

        # Fastron
        updates=50000
        support=50000
        for g in 1 2 5 10
        do
            for b in 1 10 500 1000
            do
                echo "Fastron: support ${support}, updates ${updates}, g ${g}, b ${b}"
                python compare_models.py --model_name 'Fastron' --forward_kinematics_kernel \
                    --num_training_samples $num_training_samples \
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
                    echo "DL: freq ${freq}, b ${b}, sigma ${sigma}"
                    python compare_models.py --model_name 'DL' --forward_kinematics_kernel \
                        --num_training_samples $num_training_samples \
                        --dataset_name $dataset_name \
                        --bias $b --num_freq $freq --sigma $sigma \
                        --lr 1e-3 --batch_size 512 --train_percent 0.95 --epochs 50 \
                        --results_folder $results_folder
                done
            done
        done
    done
done