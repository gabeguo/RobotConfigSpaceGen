python pareto_chart.py --data_directory comparison_results \
    --x_metric accuracy --y_metric test_time \
    --invert_x --unit_rate_y --seeds 0 --save_location pareto_charts \
    --x_label 'Error = 1 - Accuracy' --y_label 'Time per Inference (s)'