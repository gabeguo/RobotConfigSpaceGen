# parser.add_argument('--model_name', type=str, default=DL)
# # general experimental params
# parser.add_argument('--num_training_samples', type=int, default=30000)
# parser.add_argument('--dataset_name', type=str, default="3robots_25obstacles_seed0_")
# parser.add_argument('--forward_kinematics_kernel', action='store_true')
# # fastron-specific params
# parser.add_argument('--g', type=int, default=5)
# parser.add_argument('--beta', type=int, default=1)
# parser.add_argument('--maxUpdates', type=int, default=100000)
# parser.add_argument('--maxSupportPoints', type=int, default=20000)
# # dl-specific params
# parser.add_argument('--bias', type=int, default=1)
# parser.add_argument('--num_freq', type=int, default=8)
# parser.add_argument('--sigma', type=float, default=1)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--batch_size', type=int, default=512)
# parser.add_argument('--train_percent', type=float, default=0.95)
# parser.add_argument('--epochs', type=int, default=50)
# # where to log output
# parser.add_argument('--results_folder', type=str, default='comparison_results')

num_training_samples=30000
results_folder='comparison_results'
# use forward kinematics kernel!

for seed in 0 1 2
do
    dataset_name="3robots_25obstacles_seed${seed}_"

    # Fastron
    for support in 3000 10000 30000
    do
        for updates in 5000 30000
        do
            for g in 1 5 10
            do
                for b in 1 500 1000
                do
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
    for freq in 0 4 8 12
    do
        for b in 1 1.5 2
        do
            for sigma in 0.5 1 2
            do
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