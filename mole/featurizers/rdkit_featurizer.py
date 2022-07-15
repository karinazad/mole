import json
import logging
import os
from importlib import reload
from typing import Union

import deepchem as dc
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mole.featurizers.base_featurizer import Featurizer
from mole.utils.data_utils import replace_infs_with_nans, random_sample, check_convert_single_sample

from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class RDKitFeaturizer(BaseEstimator, TransformerMixin, Featurizer):
    def __init__(self):
        super(RDKitFeaturizer, self).__init__(featurizer_name="RDKitFeaturizer")

        self.featurizer = dc.feat.RDKitDescriptors()
        self.feature_names = None
        self.feature_indices = None

        self.scaler_imputer = Pipeline([
                ("imputer", SimpleImputer(missing_values=np.nan, strategy='mean')),
                ("scaler", StandardScaler()),
            ])

    def fit(self,
            smiles: Union[list, np.ndarray, pd.DataFrame, pd.Series],
            y=None,
            num_samples=None,
            random_state=0,
            **kwargs):

        if num_samples:
            smiles = random_sample(smiles, num_samples, random_state)

        X = self.featurizer(smiles)
        self.feature_names = [feature for feature in self.featurizer.descriptors if not feature.startswith('fr_')]
        self.feature_indices = [i for i, feature in enumerate(self.featurizer.descriptors)
                                if not feature.startswith('fr_')]

        X = pd.DataFrame(X[:, self.feature_indices], columns=self.feature_names)
        X = replace_infs_with_nans(X)
        self.scaler_imputer.fit(X)

        return self

    def transform(self,
                  smiles: Union[pd.DataFrame, pd.Series],
                  y=None,
                  **kwargs):

        smiles = check_convert_single_sample(smiles)

        X = self.featurizer(smiles)
        X = pd.DataFrame(X[:, self.feature_indices], columns=self.feature_names)

        X = replace_infs_with_nans(X)
        X = self.scaler_imputer.transform(X)
        X = pd.DataFrame(X[:, self.feature_indices], columns=self.feature_names, index=smiles.index)

        return X

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        params = {
            "feature_names": self.feature_names,
            "feature_indices": self.feature_indices
        }

        with open(os.path.join(save_dir, "params.json"), 'w') as fp:
            json.dump(params, fp, indent=4)

        joblib.dump(self.scaler_imputer, os.path.join(save_dir, "scaler_imputer.pkl"))

    def restore(self, save_dir):
        assert "params.json" in os.listdir(save_dir)
        assert "scaler_imputer.pkl" in os.listdir(save_dir)

        with open(os.path.join(save_dir, "params.json")) as f:
            params = json.load(f)

        self.scaler_imputer = joblib.load(os.path.join(save_dir, "scaler_imputer.pkl"))
        self.feature_names = params["feature_names"]
        self.feature_indices = params["feature_indices"]