import json
import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold


class Datagen():
    """Iterator class to sample the dataset. Tensors T_dat and E_dat are provided at runtime.

    Args:
        maxsteps: length of generator
        batchsize: samples per batch
        T_dat: transcriptomic data matrix
        E_dat: electrophysiological data matrix
    """

    def __init__(self, maxsteps, batchsize, T_dat, E_dat):
        self.T_dat = T_dat
        self.E_dat = E_dat
        self.batchsize = batchsize
        self.maxsteps = maxsteps
        self.n_samples = self.T_dat.shape[0]
        self.count = 0
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.maxsteps:
            self.count = self.count+1
            ind = np.random.randint(0, self.n_samples, self.batchsize)
            return (self.T_dat[ind, :], 
                    self.E_dat[ind, :])
        else:
            raise StopIteration


def load_bioarxiv_dataset(data_path):
    """Loads patch-seq data used in the Bioarxiv manuscript 3708 cells: 1252 genes, and 68 (44+24) ephys features.

    Args:
        data_path (str): path to data file

    Returns:
        data (dict)
    """
    data = sio.loadmat(data_path + 'PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat', squeeze_me=True)
    with open(data_path + 'E_names.json') as f:
        ephys_names = json.load(f)
    data['E_pcipfx'] = np.concatenate([data['E_pc_scaled'], data['E_feature']], axis=1)
    data['pcipfx_names'] = np.concatenate([data['pc_name'],data['feature_name']])
    temp = [ephys_names[f] for f in data['pcipfx_names']]
    data['pcipfx_names'] = np.array(temp)
    return data


def partitions(celltype, n_partitions, seed=0):
    """Create stratified cross validation sets, based on `cluster` annotations. 
    Indices of the n-th fold are `ind_dict[n]['train']` and `ind_dict[n]['val']`.
    Assumes `celltype` has the same samples as in the dataset. 

    Args:
        celltype: numpy array with celltype annotations for all dataset samples
        n_partitions: number of partitions (e.g. cross validation folds)
        seed: random seed used for partitioning.
    
    Returns:
        ind_dict: list with `n_partitions` dict elements. 
    """
    import warnings
    warnings.filterwarnings("ignore",category=UserWarning)

    #Safe to ignore warning - there are celltypes with a low sample number that are not crucial for the analysis.
    with warnings.catch_warnings():    
        skf = StratifiedKFold(n_splits=n_partitions, random_state=0, shuffle=True)

    #Get all partition indices from the sklearn generator:
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in
                skf.split(X=np.zeros(shape=celltype.shape), y=celltype)]
    return ind_dict
