import json
from constants import *

"""
This script exists because I forgot to log the simulation time
when I ran predict_points with a few more test sizes.

Merges results from sample_percent_results.json and sample_size_results_redone.json
by taking SIMULATION_TIME and SAMPLE_SIZE from sample_percent_results.json
and adding them to the contents of sample_size_results_redone.json
and creating a new file called merged_sample_size_results.json
"""

OLD_RESULTS_FNAME = 'sample_percent_results.json'
EXPANDED_RESULTS_FNAME = 'sample_size_results_redone.json'
NEW_RESULTS_FNAME = 'merged_sample_size_results.json'

if __name__ == "__main__":
    with open(OLD_RESULTS_FNAME, 'r') as fin_old, \
            open(EXPANDED_RESULTS_FNAME, 'r') as fin_expanded, \
            open(NEW_RESULTS_FNAME, 'w') as fout:
        old_results = json.load(fin_old)
        expanded_results = json.load(fin_expanded)

        curr_simulation_time = old_results["0.9"][SIMULATION_TIME]
        curr_num_points = old_results["0.9"][SAMPLE_SIZE]

        new_results = {**expanded_results}
        for percentage in new_results:
            new_results[percentage][SIMULATION_TIME] = curr_simulation_time
            new_results[percentage][SAMPLE_SIZE] = curr_num_points

        json.dump(new_results, fout)

    print(json.dumps(new_results, indent=4))
    # return
