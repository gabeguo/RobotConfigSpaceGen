for seed in {0..2}
do
    python generate_environment.py \
    --num_samples 50000 \
    --num_surface_samples 50000 \
    --seed $seed \
    --num_obstacles 25 \
    --num_robots 3 \
    --min_robot_robot_distance 2.0 \
    --max_robot_robot_distance 5.0 \
    --keyword_name 'surface_sampling' \
    --data_folder 'surface_sampling' \
    --indent_ratio 0.15
done