for seed in {0..2}
do
    python generate_environment.py \
    --num_samples 50000 \
    --num_surface_samples 50000 \
    --seed $seed \
    --num_obstacles 10 \
    --num_robots 1 \
    --keyword_name 'surface_sampling' \
    --data_folder 'surface_sampling'
done