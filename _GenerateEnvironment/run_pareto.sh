python pareto_chart.py --data_directory comparison_results \
    --x_metric accuracy --y_metric test_time \
    --invert_x --unit_rate_y --seeds 0 1 2 --save_location pareto_charts \
    --x_label 'Error = 1 - Accuracy' --y_label 'Time per Inference (s)' \
    --title 'Error = (1 - Accuracy) vs. Time per Inference (s):\n21 DoF, 30K Train, 5K Test'

python pareto_chart.py --data_directory comparison_results \
    --x_metric tpr --y_metric tnr \
    --invert_x --invert_y --seeds 0 1 2 --save_location pareto_charts \
    --x_label 'Error = 1 - TPR' --y_label 'Error = 1 - TNR' \
    --title 'TPR vs. TNR: \n21 DoF, 30K Train, 5K Test'