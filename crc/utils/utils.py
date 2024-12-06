import json
import numpy as np
from sklearn.model_selection import train_test_split


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


def train_val_test_split(*arrays, train_size, val_size=None, test_size=None, random_state):
    train_out, intermed_out = train_test_split(*arrays, train_size=train_size,
                                               random_state=random_state)

    try:
        val_out, test_out = train_test_split(intermed_out,
                                             train_size=(val_size/(1-train_size)),
                                             random_state=random_state)
    except TypeError:
        val_out, test_out = train_test_split(intermed_out,
                                             test_size=(test_size/(1-train_size)),
                                             random_state=random_state)

    return train_out, val_out, test_out

