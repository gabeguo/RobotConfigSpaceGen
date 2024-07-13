python pareto_chart.py --data_directory ablation_results \
    --x_metric accuracy --y_metric accuracy \
    --seeds 0 1 2 --save_location ablation_charts \
    --plot_bar_chart \
    --title '21 DoF, 30K Train, 5K Test'

python pareto_chart.py --data_directory ablation_results \
    --x_metric tpr --y_metric tpr \
    --seeds 0 1 2 --save_location ablation_charts \
    --plot_bar_chart \
    --title '21 DoF, 30K Train, 5K Test'

python pareto_chart.py --data_directory ablation_results \
    --x_metric tnr --y_metric tnr \
    --seeds 0 1 2 --save_location ablation_charts \
    --plot_bar_chart \
    --title '21 DoF, 30K Train, 5K Test'