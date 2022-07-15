import logging
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from mole.models.base_model import BaseModel
from mole.utils.data_utils import convert_array_to_series
from mole.utils.common import silence_stdout, disable_tqdm, setup_logger

disable_tqdm()
import chemprop

from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger, level="ERROR")


class ChempropModel(BaseModel):

    def __init__(self, checkpoint_dir=None, custom_logger=None, wandb_logger=None, **kwargs):
        super().__init__(**kwargs)
        if checkpoint_dir:
            arguments = [
                '--test_path', '/dev/null',
                '--preds_path', '/dev/null',
                '--checkpoint_dir', checkpoint_dir,
            ]
            with silence_stdout():
                self.args = chemprop.args.PredictArgs().parse_args(arguments)
                self.model = chemprop.train.load_model(args=self.args)

        else:
            self.args = None
            self.model = None

        if custom_logger:
            self.custom_logger = custom_logger
        else:
            self.custom_logger = logger

        self.wandb_logger = wandb_logger

    def train(self, data_path, save_dir, task_type,
              separate_test_path=None, separate_val_path=None, save_smiles_splits=None,
              epochs=None, num_folds=None, pretrained_dir=None, retrain=False,
              config_path=None):

        train_arguments = [
            '--data_path', data_path,
            '--dataset_type', task_type,
            '--save_dir', save_dir,
            "--split_type", "scaffold_balanced"
        ]

        if epochs:
            train_arguments += ["--epochs", str(epochs)]

        if num_folds is not None:
            print(f"Running cross-validated training with {num_folds} folds.")

            train_arguments += ["--num_folds", str(num_folds)]

        if save_smiles_splits:
            train_arguments += ["--save_smiles_splits"]

        if separate_val_path is not None:
            train_arguments += ["--separate_val_path", separate_val_path]

        if separate_test_path is not None:
            train_arguments += ["--separate_test_path", separate_test_path]

            if separate_val_path is None and not retrain:
                train_arguments += ["--split_sizes", "0.9", "0.1", "0.0"]

        if retrain:
            print("Retraining on all available train data")
            train_arguments += ["--split_sizes", "0.98", "0.01", "0.01"]

        if pretrained_dir is not None:
            print(f"Using pretrained model at {pretrained_dir} ")
            train_arguments += ["--checkpoint_dir", pretrained_dir]

        if config_path is not None:
            print(f"Using config  at {config_path} ")
            train_arguments += ["--config_path", config_path]

        args = chemprop.args.TrainArgs().parse_args(train_arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

        # Set the model to self.model by restoring it from checkpoints
        self.restore(save_dir)
        assert self.model, "Failed to load the model from checkpoints."

        return mean_score, std_score

    def predict(self, X):

        with silence_stdout():
            assert self.model is not None, "Please initialize the model by providing a valid checkpoint"

            index = X.index
            X = np.array(X).reshape(-1, 1)
            X = X.tolist()
            y_pred = chemprop.train.make_predictions(smiles=X, args=self.args, model_objects=self.model)
            y_pred = np.array(y_pred).flatten()

            # Try to return the predictions as pd.Series else return the default numpy array
            y_pred = convert_array_to_series(y_pred, index)

        return y_pred

    def evaluate(self, X_test, y_test, metric=None):
        y_pred = self.predict(X_test)
        return self._evaluate(y_test, y_pred, metric)

    def save(self, save_dir):
        raise NotImplementedError

    def restore(self, save_dir):

        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', save_dir,
        ]

        with silence_stdout():
            self.args = chemprop.args.PredictArgs().parse_args(arguments)
            self.model = chemprop.train.load_model(args=self.args)
