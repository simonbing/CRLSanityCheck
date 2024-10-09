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
    elif task == 'lt_pcl_1':
        exp_name = 'ar_1_uniform'
        env_list = ['ref']

    return exp_name, env_list
