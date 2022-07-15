import logging
from abc import abstractmethod
import numpy as np
import sklearn

from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class BaseModel:
    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        if self.mode == "classification":
            assert kwargs.get("num_classes", None) is not None, \
                "Please provide the number of classes for classification"

        self.num_classes = kwargs.get("num_classes", None)
        self.model_dir = None
        self.custom_logger = kwargs.get("custom_logger", None)

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, metric="RMSE"):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def _evaluate(self, y_test, y_pred, metric):
        y_test, y_pred = np.array(y_test).flatten(), np.array(y_pred).flatten()
        assert y_test.shape == y_pred.shape

        if metric is None:
            if self.mode == "regression":
                metric = "RMSE"
            else:
                metric = "AUC"

        if metric == "RMSE":
            score = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)
        elif metric == "MSE":
            score = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=True)
        elif metric == "AUC":
            score = sklearn.metrics.roc_auc_score(y_test, y_pred)

        else:
            raise NotImplementedError

        return score

    @abstractmethod
    def save(self, save_dir):
        pass

    @abstractmethod
    def restore(self, save_dir):
        pass

    def _log(self, msg, level="info"):
        if self.custom_logger is not None:
            if level == "info":
                self.custom_logger.info(msg)
            elif level == "warning":
                self.custom_logger.warning(msg)
            elif level == "error":
                self.custom_logger.error(msg)
            elif level == "exception":
                self.custom_logger.exception(msg)
        else:
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "exception":
                logger.exception(msg)

