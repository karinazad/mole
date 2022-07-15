import logging
import math

import numpy as np
import pandas as pd
import deepchem as dc
import rdkit.Chem as Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.cluster import AgglomerativeClustering

from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


def replace_infs_with_nans(X):
    if type(X) == np.ndarray:
        X[X == -np.inf] = np.nan
        X[X == np.inf] = np.nan

    elif type(X) == pd.DataFrame or type(X) == pd.Series:
        X = X.fillna(np.nan)
        X = X.replace([np.inf, -np.inf, math.inf, -math.inf], np.nan)

    return X


def random_sample(array, samples, random_state=0):
    np.random.seed(random_state)
    mask = np.random.randint(0, len(array), samples)

    if type(array) == pd.Series or type(array) == pd.DataFrame:
        sampled_array = array.iloc[mask]

    else:
        sampled_array = array[mask]

    return sampled_array


def check_convert_single_sample(inputs):
    if type(inputs) == str:
        inputs = [inputs]

    return inputs


def drop_empty_arrays(X):
    indices_to_drop = [i for i in range(len(X)) if type(X[i]) == np.ndarray]
    X_ = np.delete(X, indices_to_drop, axis=0)

    return X_, indices_to_drop


def convert_array_to_series(array, indices):
    result = array
    try:
        result = pd.Series(array, index=indices)
    except Exception as e:
        try:
            result = pd.Series(array.flatten(), index=indices)
        except:
            logger.warning("Warning: forced to return predictions in a numpy array."
                           " The output predictions might not correspond to the input's order.")

    return result


def convert_input_to_array(X):
    if type(X) == pd.DataFrame or type(X) == pd.Series:
        X = X.values
    elif type(X) == np.ndarray:
        pass
    else:
        error_msg = f"Input type: {type(X)} not recognized"
        logger.exception(error_msg)
        raise NotImplementedError(error_msg)

    return X


def check_dc_graph_inputs(X, y=None):
    graph_mask = [type(x) is dc.feat.graph_data.GraphData for x in X]
    failed_indices = [np.where(~np.array(graph_mask, dtype=bool))[0].tolist()]
    logger.info(f"Indices of failed inputs: {failed_indices}")

    if sum(graph_mask) == 0:
        error_msg = "Invalid inputs provided (expected dc.feat.graph_data.GraphData). Most likely, the problem" \
                    "occurred during featurization."
        logger.error(error_msg)
        raise UserWarning(error_msg)

    X = X[graph_mask]

    if y is not None:
        y = y[graph_mask]
        assert len(X) == len(y)
        return X, y

    else:
        return X


def get_distance_matrix(dataset):
    all_fps = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), radius=2, nBits=1024) for mol in dataset.ids]

    N = len(all_fps)
    SIM_MAT = np.zeros((N, N))
    for i in range(N):
        SIM_MAT[i, i + 1:] = BulkTanimotoSimilarity(all_fps[i], all_fps[i + 1:])

    SIM_MAT = SIM_MAT + SIM_MAT.T
    for i in range(N):
        SIM_MAT[i, i] = 1

    DIST_MAT = 1 - SIM_MAT
    return DIST_MAT


def get_clusters(dist_mat, dist_th):
    model = AgglomerativeClustering(distance_threshold=dist_th, n_clusters=None, affinity="precomputed",
                                    linkage="single")
    clusters = model.fit_predict(dist_mat)
    cluster_counts = np.unique(clusters, return_counts=True)[1]
    return clusters, cluster_counts


def to_float(x):
    try:
        return float(x)
    except:
        return np.nan
