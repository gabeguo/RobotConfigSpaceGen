from common_functions import *
from constants import *
import argparse

# for each combo of model + dataset,
# results are averaged over all environments with such model + dataset combo
COMPARISON_VARIABLES = {
    TRAIN_IDX_LOW,
    TRAIN_IDX_HIGH,
    TEST_IDX_LOW,
    TEST_IDX_HIGH,
    'model_name',
    'forward_kinematics_kernel',
    'g',
    'beta',
    'maxUpdates',
    'maxSupportPoints',
    'bias',
    'num_freq',
    'sigma',
    'lr',
    'batch_size',
    'train_percent',
    'epochs'
}

X_VAR_KEY = [TRAIN_IDX_LOW, TRAIN_IDX_HIGH, TEST_IDX_LOW, TEST_IDX_HIGH]

TRAIN_RANGES = {
    (20000, 50000):'Rejection', 
    (35000, 65000):'Combo',
    (50000, 80000):'Uniform',
}

TEST_RANGES = {
    # (0, 5000):'Surface',
    (95000, 100000):'Uniform'
}

X_VAL_TO_LABEL = {
    tuple(list(key_train) + list(key_test)):f"{TRAIN_RANGES[key_train]},\n{TEST_RANGES[key_test]}"
    for key_train in TRAIN_RANGES for key_test in TEST_RANGES
}

def label_plot(args, df_mean_std):
    plt.xlabel('Sampling Scheme (Train, Test)')
    #plt.xticks(unique_x_values_list)
    metric_name = args.metric.capitalize() if args.metric not in [TPR, TNR] else args.metric.upper()
    if args.ylabel:
        plt.ylabel(args.ylabel)
    else:
        plt.ylabel(metric_name)
    plt.legend(bbox_to_anchor=(0, -0.28, 1, -0.02), loc="lower left",
        mode="expand", borderaxespad=0, ncol=3, fontsize='small')
    plt.subplots_adjust(bottom=0.2)
    
    # if args.metric != TPR:
    #     plt.ylim(0.4, 1)
    #     plt.yticks(np.linspace(0.4, 1, 13))
    # else:
    #     plt.ylim(0, 1)
    #     plt.yticks(np.linspace(0, 1, 11))

    num_training_samples = df_mean_std['train_idx_high'][0] - df_mean_std['train_idx_low'][0]
    num_testing_samples = df_mean_std['test_idx_high'][0] - df_mean_std['test_idx_low'][0]

    plt.title(f'Impact of Sampling Scheme on Model Performance:' + 
              f'\n21 DoF, {num_training_samples} Train, {num_testing_samples} Test')
    plt.savefig(f'{args.save_location}/Sampling Scheme {metric_name}.pdf')
    plt.savefig(f'{args.save_location}/Sampling Scheme {metric_name}.png')
    #plt.show()
    return

def main(args):
    plt.rcParams.update({'figure.figsize': (8, 6)})
    plt.rcParams.update({'font.size': 11})

    # get all the data points
    df_mean_std, baseline, DOF, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES \
        = load_json_files_pd(args=args, COMPARISON_VARIABLES=COMPARISON_VARIABLES,
                            the_x_var_key=X_VAR_KEY, 
                            expected_num_x_vals=3, baseline_data_dir='surface_sampling') 
    result_df = plot_results(df=df_mean_std, baseline_by_level=baseline, y_values=df_mean_std[(args.metric, 'mean')].tolist(), 
                the_x_var_key=X_VAR_KEY, num_test_samples=NUM_TEST_SAMPLES, 
                expected_unique_x_val_length=3, args=args, x_val_to_label=X_VAL_TO_LABEL,
                all_model_names=[DL_CUDA, FASTRON], return_df=True)

    plt.clf()
    # https://stackoverflow.com/a/25449186
    ax = result_df.plot(x=SAMPLING_SCENARIO_KEY, kind='bar', stacked=False, rot=0, width=0.9,
                        color=['tab:orange', 'tab:green', 'tab:blue'])
    hatches = ['x', 'x', 'x', '+', '+', '+', 'o', 'o', 'o']
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + 0.05, p.get_height() + 0.01))
        p.set_hatch(hatches.pop())
    print(result_df)

    label_plot(args, df_mean_std)

    return

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("--data_directory", type=str, default='surfaceSampling_experiment_results')
    parser.add_argument("--metric", type=str, default='accuracy')
    parser.add_argument('--invert_metric', action='store_true', help='min metric is best value', default=False)
    parser.add_argument('--unit_rate_metric', action='store_true', help='Divide metric by number of samples', default=False)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--save_location", type=str, default='graphs')
    parser.add_argument("--disable_error_bars", action='store_true')
    parser.add_argument("--include_gpu", action='store_true')
    parser.add_argument("--plot_medians", action='store_true')

    # Execute the parse_args() method
    args = parser.parse_args()

    os.makedirs(args.save_location, exist_ok=True)

    main(args)
