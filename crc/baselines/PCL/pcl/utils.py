""" Utilities
"""


import numpy as np
import scipy as sp
import os
import shutil
import tarfile

from subfunc.showdata import *
from subfunc.munkres import Munkres

from causalchamber.datasets import Dataset as ChamberDataset

from crc.utils import get_task_environments


# =============================================================
# =============================================================
def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
         method: correlation method ('Pearson' or 'Spearman')
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
     """

    print('Calculating correlation...')

    x = x.copy().T
    y = y.copy().T
    dimx = x.shape[0]
    dimy = y.shape[0]

    # Calculate correlation -----------------------------------
    if method == 'Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dimy, dimy:]
    elif method == 'Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dimy, dimy:]
    else:
        raise ValueError

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dimy, dtype=int)
    for i in range(dimy):
        sort_idx[i] = indexes[i][1]
    sort_idx_other = np.setdiff1d(np.arange(0, dimx), sort_idx)
    sort_idx = np.concatenate([sort_idx, sort_idx_other])

    x_sort = x[sort_idx, :]

    # Re-calculate correlation --------------------------------
    if method == 'Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dimy, dimy:]
    elif method == 'Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dimy, dimy:]
    else:
        raise ValueError

    return corr_sort, sort_idx, x_sort


# ===============================================================
# ===============================================================
def unzip(loadfile, unzipfolder, necessary_word='/storage'):
    """

    unzip trained model (loadfile) to unzipfolder

    """

    print('load: %s...' % loadfile)
    if loadfile.find(".tar.gz") > -1:
        if unzipfolder.find(necessary_word) > -1:
            if os.path.exists(unzipfolder):
                print('delete savefolder: %s...' % unzipfolder)
                shutil.rmtree(unzipfolder)  # remove folder
            archive = tarfile.open(loadfile)
            archive.extractall(unzipfolder)
            archive.close()
        else:
            assert False, "unzip folder doesn't include necessary word"
    else:
        if os.path.exists(unzipfolder):
            print('delete savefolder: %s...' % unzipfolder)
            shutil.rmtree(unzipfolder)  # remove folder
        os.makedirs(unzipfolder)
        src_files = os.listdir(loadfile)
        for fn in src_files:
            full_file_name = os.path.join(loadfile, fn)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, unzipfolder + '/')
        # unzipfolder = loadfile

    if not os.path.exists(unzipfolder):
        raise ValueError


def get_chamber_data_PCL(dataset, data_root, task):
    exp, env_list = get_task_environments(task)

    chamber_data = ChamberDataset(dataset, root=data_root, download=True)

    data = chamber_data.get_experiment(name=f'{exp}_reference').as_pandas_dataframe()

    features = ['red', 'green', 'blue', 'pol_1', 'pol_2']  # hardcoded for now

    x_gt = data[features].to_numpy()


