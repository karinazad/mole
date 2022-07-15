from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class Featurizer(ABC):
    def __init__(self, featurizer_name):
        self.failed_indices = []
        self.featurizer_name = featurizer_name

    @abstractmethod
    def fit(self, smiles: Union[pd.DataFrame, pd.Series, list, np.ndarray], y=None, **kwargs):
        pass

    @abstractmethod
    def transform(self, smiles: Union[pd.DataFrame, pd.Series], y=None, **kwargs):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass

    @abstractmethod
    def restore(self, save_dir):
        pass