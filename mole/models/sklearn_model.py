import json
import logging
import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from .base_model import BaseModel

from mole.utils.data_utils import convert_array_to_series
from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class SklearnModel(BaseEstimator, BaseModel):
    def __init__(self, model=None, **kwargs):
        super(SklearnModel, self).__init__(**kwargs)

        self.model = model

    def fit(self,  X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X: pd.Series):
        if self.mode == "classification":
            if self.num_classes == 2:
                y_pred = self.model.predict_proba(X)[:, 1]
            else:
                y_pred = self.model.predict_proba(X)
        else:
            y_pred = self.model.predict(X)

        # Try to return the predictions as pd.Series else return the default numpy array
        y_pred = convert_array_to_series(y_pred, X.index)

        return y_pred

    def evaluate(self, X_test, y_test, metric="RMSE"):
        y_pred = self.predict(X_test)
        return self._evaluate(y_test, y_pred, metric)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        params = {
            "mode": self.mode,
            "num_classes": self.num_classes,
        }

        with open(os.path.join(save_dir, "params.json"), 'w') as fp:
            json.dump(params, fp, indent=4)

        joblib.dump(self.model, os.path.join(save_dir, "model.pkl"))

    def restore(self, save_dir):
        assert "params.json" in os.listdir(save_dir)
        assert "model.pkl" in os.listdir(save_dir)

        with open(os.path.join(save_dir, "params.json")) as f:
            params = json.load(f)

        self.mode = params["mode"]
        self.num_classes = params["num_classes"]
        self.model = joblib.load(os.path.join(save_dir, "model.pkl"))
