import logging
from typing import Union
import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from mole.featurizers.base_featurizer import Featurizer
from mole.utils.data_utils import check_convert_single_sample, convert_array_to_series
from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class AttentiveFPFeaturizer(BaseEstimator, TransformerMixin, Featurizer):
    def __init__(self, **kwargs):
        super(AttentiveFPFeaturizer, self).__init__(featurizer_name="AttentiveFPFeaturizer")
        self.featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=kwargs.get("use_edges", True), **kwargs)

    def fit(self, smiles: Union[list, np.ndarray, pd.DataFrame, pd.Series], y=None, **kwargs):
        return self

    def transform(self, smiles: Union[pd.DataFrame, pd.Series], y=None, **kwargs):
        self.failed_indices = []
        smiles = check_convert_single_sample(smiles)
        X = self.featurizer.featurize(smiles)

        X = convert_array_to_series(X, smiles.index)
        return X

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
