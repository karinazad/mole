import json
import logging
import os
import deepchem as dc
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from mole.models.base_model import BaseModel
from mole.utils.data_utils import convert_array_to_series, check_dc_graph_inputs
from mole.utils.common import is_jsonable

from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class DeepChemModel(BaseEstimator, BaseModel):
    def __init__(self,
                 model=None,
                 mode="regression",
                 **kwargs):

        self.init_params = kwargs

        super(DeepChemModel, self).__init__(mode=mode,
                                            num_classes=kwargs.get("num_classes", None),
                                            custom_logger=kwargs.get("custom_logger", None),
                                            )

        if model == "AttentiveFP":
            self.model = dc.models.AttentiveFPModel(mode=mode,
                                                    n_tasks=1,
                                                    n_classes=kwargs.get("num_classes", None),
                                                    **kwargs)

        else:
            self.model = model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        self.fit_params = kwargs
        self._log(f"Number of training samples = {len(X_train)}.")

        X_train, y_train = check_dc_graph_inputs(X_train, y_train)
        self._log(f"Number of correctly featurized samples = {len(X_train)}.")

        indices = X_train.index if hasattr(X_train, "index") else np.arange(len(X_train))
        train_dataset = dc.data.NumpyDataset(X=X_train, y=y_train, ids=indices)

        if kwargs.pop("balance_classes", None):
            transformer = dc.trans.BalancingTransformer(dataset=train_dataset)
            train_dataset = transformer.transform(train_dataset)
            self._log(f"Training data processed with BalancingTransformer")

        if X_val is not None and y_val is not None:

            X_val, y_val = check_dc_graph_inputs(X_val, y_val)
            indices = X_val.index if hasattr(X_val, "index") else np.arange(len(X_val))
            val_dataset = dc.data.NumpyDataset(X=X_val, y=y_val, ids=indices)

            metric = kwargs.pop("metric", None)

            if metric is None:
                if self.mode == "classification":
                    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode=self.mode)
                else:
                    metric = dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean, mode=self.mode)

            self._log(f"Validation set included with metric {metric}")
            callback = dc.models.ValidationCallback(val_dataset, 100, metric)

        else:
            callback = []

        return self.model.fit(train_dataset, callbacks=callback, **kwargs)

    def predict(self, X: pd.Series):

        self._log(f"Number of inference samples = {len(X)}.")

        X = check_dc_graph_inputs(X)
        self._log(f"Number of correctly featurized samples = {len(X)}.")

        indices = X.index if hasattr(X, "index") else np.arange(len(X))

        Xdc = dc.data.NumpyDataset(np.array(X), ids=indices)
        y_pred = self.model.predict(Xdc)

        if self.mode == "classification" and self.num_classes == 2:
            y_pred = y_pred[:, 1]

        # Try to return the predictions as pd.Series else return the default numpy array
        y_pred = convert_array_to_series(y_pred, X.index)

        return y_pred

    def evaluate(self, X_test, y_test, metric=None):
        X_test, y_test = check_dc_graph_inputs(X_test, y_test)
        y_pred = self.predict(X_test)
        score = self._evaluate(y_test, y_pred, metric)

        self._log(f"{metric} score = {score}")

        return score

    def save(self, save_dir):
        assert self.model is not None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        params = {
            "mode": self.mode,
            "num_classes": self.num_classes,
            "dc_model_class": type(self.model).__name__,
        }

        with open(os.path.join(save_dir, "params.json"), 'w') as fp:
            json.dump(params, fp, indent=4)

        self.model.save_checkpoint(model_dir=save_dir)

        with open(os.path.join(save_dir, f"verbose_config.json"), 'w') as f:
            config = {}
            try:
                config.update(self.init_params)
                config.update(self.fit_params)
            except AttributeError:
                pass

            config.update({str(k): str(v) for k, v in self.__dict__.items() if is_jsonable(str(k), str(v))})
            config.update({str(k): str(v) for k, v in self.model.__dict__.items() if is_jsonable(str(k), str(v))})

            json.dump(config, f)

        self._log(f"Model saved  at {save_dir}")

    def restore(self, save_dir):
        self._log(f"Loading model from {save_dir}...")

        assert "params.json" in os.listdir(save_dir)

        with open(os.path.join(save_dir, "params.json")) as f:
            params = json.load(f)

        self.mode = params["mode"]
        assert self.mode in ["classification", "regression"]

        if self.mode == "classification":
            self.num_classes = params["num_classes"]

        dc_model_class = params["dc_model_class"]

        if dc_model_class == "AttentiveFPModel":
            self.model = dc.models.AttentiveFPModel(mode=self.mode, n_tasks=1)
        else:
            self._log(f"{dc_model_class} is not supported", level="ERROR")
            raise NotImplementedError

        # Get saved model checkpoints
        checkpoints = [x for x in os.listdir(save_dir) if x.startswith("checkpoint")]

        if not checkpoints:
            error_msg = f"No model checkpoints found at {save_dir}"
            self._log(error_msg, level="ERROR")
            raise FileNotFoundError(error_msg)

        if "checkpoint.pt" in checkpoints:
            # Assume that checkpoint.pt  has a priority over numbered checkpoints
            checkpoint = "checkpoint.pt"
        else:
            # Pick the latest checkpoint
            checkpoint = sorted(checkpoints)[-1]

        data = torch.load(os.path.join(save_dir, checkpoint), map_location=torch.device('cpu'))
        self.model.model.load_state_dict(data['model_state_dict'])

        self._log(f"Model successfully loaded.")
