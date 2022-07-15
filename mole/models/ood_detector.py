import contextlib
import io
import json
import logging
import os

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mole.models.base_model import BaseModel
from mole.utils.data_utils import convert_input_to_array, convert_array_to_series
from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)

class OODDetector(BaseModel):
    def __init__(self, model_selection=None, **kwargs):

        # self.model_selection = model_selection
        # For now, only implements the Normalizing Flow model
        super().__init__(**kwargs)
        self.model_selection = {"Flow"}

    def _hyperparameter_search(self, X_train, X_val):
        try:
            import selecting_OOD_detector
        except:
            raise UserWarning("This module requires \"selecting_OOD_detector\" package")

        hyperparm_tuner = selecting_OOD_detector.HyperparameterTuner(num_evals_per_model=5, model_selection=self.model_selection)
        hyperparm_tuner.run_hyperparameter_search(X_train=X_train,
                                                  X_val=X_val,
                                                  y_train=None,
                                                  y_val=None,
                                                  save_intermediate_scores=False,
                                                  )
        model_params = {
            "init": hyperparm_tuner.get_best_parameteres(),
            "train": hyperparm_tuner.train_params
        }
        return model_params

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        try:
            import selecting_OOD_detector
        except:
            raise UserWarning("This module requires \"selecting_OOD_detector\" package")


        X_train, X_test = train_test_split(X_train, random_state=0)

        if X_val is None:
            X_train, X_val = train_test_split(X_train, random_state=0)

        self.scaler = StandardScaler()

        with contextlib.redirect_stdout(io.StringIO()):
            self.model_params = self._hyperparameter_search(X_train, X_val)

        pipe = selecting_OOD_detector.OODPipeline(model_selection=self.model_selection)
        pipe.fit(X_train, X_test=X_test, hyperparameters=self.model_params)

        self.model = pipe.novelty_estimators[0]["Flow"]

        self.test_scores = self.predict_likelihood(X_test)
        self.scaler.fit(self.test_scores.reshape(-1, 1))

    def predict(self, X):
        pred = self.predict_likelihood(X)
        pred = self.scaler.transform(pred.reshape(-1, 1))
        # pred = sigmoid(- pred)
        pred = convert_array_to_series(pred, X.index)

        return pred

    def predict_likelihood(self, X):
        X = convert_input_to_array(X)
        pred = self.model.get_novelty_score(X)
        return pred

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        params = {
            # "test_scores": list(self.test_scores),
            "model_params": self.model_params,
        }

        with open(os.path.join(save_dir, "params.json"), 'w') as fp:
            json.dump(params, fp, indent=4)

        joblib.dump(self.model, os.path.join(save_dir, "model.pkl"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))

    def restore(self, save_dir):
        assert "scaler.pkl" in os.listdir(save_dir)
        assert "model.pkl" in os.listdir(save_dir)
        assert "params.json" in os.listdir(save_dir)

        with open(os.path.join(save_dir, "params.json")) as f:
            params = json.load(f)

        # self.test_scores = params.get("test_scores")
        self.model_params = params.get("model_params")
        self.model = joblib.load(os.path.join(save_dir, "model.pkl"))
        self.scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
