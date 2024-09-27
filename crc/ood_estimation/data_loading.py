import pandas as pd
from sklearn.preprocessing import StandardScaler

from causalchamber.datasets import Dataset as ChamberData


def get_chamber_data(task, data_root):
    """
    Returns X as a pandas dataframe and y directly as np array.
    :param task:
    :param data_root:
    :return:
    """
    if task == 'lt_1':
        chamber_data = ChamberData(name='lt_camera_v1', root=data_root, download=True)
        features = ['red', 'green', 'blue', 'pol_1', 'pol_2', 'image_file']
        target = 'ir_1'

        # In-distribution training environments
        envs_list_id = ['reference', 'red', 'green', 'blue', 'pol_1']
        df_list_id = []
        y_df_list_id = []
        for env in envs_list_id:
            env_df = chamber_data.get_experiment(name=f'scm_2_{env}').as_pandas_dataframe()
            env_df['image_file'] =[f'lt_camera_v1/scm_2_{env}/images_64/{item}' for item in env_df['image_file']]

            df_list_id.append(env_df[features])
            y_df_list_id.append(env_df[target])

        # Out-of-distribution test environment
        env_ood = 'pol_2'
        df_ood = chamber_data.get_experiment(name=f'scm_2_{env_ood}').as_pandas_dataframe()
        df_ood['image_file'] = [f'lt_camera_v1/scm_2_{env_ood}/images_64/{item}' for item in df_ood['image_file']]

    X_df_train = pd.concat(df_list_id)
    y_train = pd.concat(y_df_list_id).to_numpy()

    X_df_test = df_ood[features]
    y_test = df_ood[target].to_numpy()

    # Scale labels
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))

    y_train = scaler.transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))

    return X_df_train, X_df_test, y_train, y_test
