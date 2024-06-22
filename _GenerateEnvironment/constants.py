ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1 = 'f1'
ROC_AUC = 'roc_auc'
TPR = 'tpr'
TNR = 'tnr'

TRAIN_TIME = 'train_time'
TEST_TIME = 'test_time'
TRAIN_SIZE = 'train_size'
TEST_SIZE = 'test_size'
TOTAL_TIME = 'total_time'

SIMULATION_TIME = TIME_COST = 'simulation_time'
COLLISION_TIME = 'collision_detection_time'
DISTANCE_TIME = 'distance_calculation_time'
FK_QUERY_TIME = 'fk_query_time'
SAMPLE_SIZE = 'num_points'

PERCENT_COLLISION = 'collision_ratio'

DOF_KEY = 'DOF'
DOF_PER_ROBOT = 7

SIMULATION = 'PyBullet Simulation'

XGBOOST = 'XGBoost'
KNN = 'KNN'
DL = 'DL'
DUMMY = 'Majority Rule'
FASTRON = 'Fastron'

MS_PER_SEC = 1000

GRAPH_FOLDER_NAME = 'graphs'

DATA_FOLDER = 'simulation_data'
RESULTS_FOLDER = 'approximation_results'

TP_NAME = 'true positive'
TN_NAME = 'true negative'
FP_NAME = 'false positive'
FN_NAME = 'false negative'

LAYER_BY_LAYER_TIME = 'layer by layer time'

DL_CUDA = f"{DL}--use_cuda"
FULL_MODEL_NAME = {DL: 'DeepCollide Seq', FASTRON: 'Fastron FK', DL_CUDA: 'DeepCollide Parallel'}

CLF_TO_MAX_MARKER = {DL: 'o', FASTRON: 'x', DL_CUDA: 's'}
CLF_TO_MEAN_MARKER = {DL: '^', FASTRON: 'v', DL_CUDA: 'D'}
CLF_TO_MAX_COLOR = {DL: (0.1, 0.8, 0.1, 1.0), FASTRON: (0.8, 0.1, 0.1, 1.0), DL_CUDA: (0.1, 0.1, 0.8, 1.0)}
CLF_TO_MEAN_COLOR = {DL: (0.2, 0.7, 0.2, 0.5), FASTRON: (0.7, 0.2, 0.2, 0.5), DL_CUDA: (0.2, 0.2, 0.7, 0.5)}

COLLISION_DENSITY_KEY = 'collision_density'