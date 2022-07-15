import os
import pandas as pd
import unittest
import numpy as np
import shutil
import logging

from mole.models import ChempropModel

logging.disable(logging.CRITICAL)

ROOT_PATH = ""


class TestChempropModel(unittest.TestCase):

    temp_save_path = "temp"

    @classmethod
    def setUp(self) -> None:

        # Prepare Data
        # Include also invalid SMILES to check whether it is successfully removed
        smiles = pd.Series([
            "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
            "CCC1(CC)C(=O)NC(=O)N(C)C1=O",
            "O=P1(N(CCCl)CCCl)NCCCO1",
            "CC(O)C(=O)O",
            "DEFINITELY NOT SMILES STRING",
            "",
            "XAVIER"
        ])
        y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.0, -0.1])

        self.df = pd.DataFrame({"SMILES": smiles, "y": y})
        self.model = ChempropModel(mode="regression", num_classes=None)


    def test_train(self):

        model_path = os.path.join(self.temp_save_path, "model")
        data_path = os.path.join(self.temp_save_path, "TestChempropModel.csv")

        os.makedirs(model_path, exist_ok=True)
        self.df.to_csv(data_path, index=None)

        train_args = {
            "data_path": data_path,
            "save_dir": model_path,
            "task_type": "regression",
            "separate_test_path": data_path,
            "separate_val_path": None,
            "epochs": 2,
            "num_folds": 2
        }
        self.model.train(**train_args)

    def evaluate(self):
        score_rmse = self.model.evaluate(self.df.SMILES, self.df.y, metric="RMSE")
        score_mse = self.model.evaluate(self.df.SMILES, self.df.y, metric="MSE")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.temp_save_path)
