import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_task_environments(task):
    """
    Returns the causal chambers experiments to include in the training data
    for a specified task.

    :param task:
    :return:
    """
    if task == 'lt_scm_2':
        exp_name = 'scm_2'
        env_list = ['red', 'green', 'blue', 'pol_1', 'pol_2']
        features = ['red', 'green', 'blue', 'pol_1', 'pol_2']
    elif task == 'lt_scm_4':
        exp_name = 'scm_4'
        env_list = ['red', 'green', 'blue']
        features = ['red', 'green', 'blue']
    elif task == 'lt_scm_5':
        exp_name = 'scm_5'
        env_list = ['red', 'green', 'blue']
        features = ['red', 'green', 'blue']
    elif task == 'lt_1':
        exp_name = 'scm_2'
        env_list = ['red', 'green', 'blue', 'pol_1']
        features = ['red', 'green', 'blue', 'pol_1', 'pol_2']
    elif task in ('lt_pcl_1', 'lt_pcl_2'):
        exp_name = 'ar_1_uniform'
        env_list = ['ref']
        features = ['red', 'green', 'blue', 'pol_1', 'pol_2']

    return exp_name, env_list, features
