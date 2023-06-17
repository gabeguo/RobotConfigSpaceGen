num_training_samples=30000
results_folder='obstacles_experiment_results'
# use forward kinematics kernel!

for num_obstacles in {10..60..10}
do
    echo "${num_obstacles} obstacles"
    for seed in 0 1 2
    do
        for environment_type in "colliding" "separate"
        do
            dataset_name="3robots_${num_obstacles}obstacles_seed${seed}_${environment_type}Robots"

            # Fastron
            for support in 3000 10000 30000
            do
                for updates in 5000 30000
                do
                    for g in 1 5 10
                    do
                        for b in 1 500 1000
                        do
                            echo "Fastron: support ${support}, updates ${updates}, g ${g}, b ${b}"
                            python compare_models.py --model_name 'Fastron' --forward_kinematics_kernel \
                                --num_training_samples $num_training_samples \
                                --dataset_name $dataset_name \
                                --g $g --beta $b --maxUpdates $updates --maxSupportPoints $support \
                                --results_folder $results_folder
                        done
                    done
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
done