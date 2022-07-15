import os
import pandas as pd
import unittest
import numpy as np
from deepchem.models import AttentiveFPModel

import logging

from mole.featurizers import AttentiveFPFeaturizer
from mole.models import DeepChemModel

logging.disable(logging.CRITICAL)

ROOT_PATH = ""


class TestDeepchemModel(unittest.TestCase):
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

        self.assertTrue(len(smiles) == len(y))

        featurizer = AttentiveFPFeaturizer()
        self.X_train, self.y_train = featurizer.transform(smiles), y
        self.X_test, self.y_test = featurizer.transform(smiles), y
        self.X_val, self.y_val = featurizer.transform(smiles), y

        self.assertTrue(len(self.X_train) == len(self.y_train))

        # Initialize model
        init_params = {
            "batch_size": 2,
            "learning_rate": 0.001
        }
        model = AttentiveFPModel(
            mode="regression",
            n_tasks=1,
            **init_params
        )
        self.dcmodel = DeepChemModel(model=model, mode="regression", num_classes=None)


    def test_fit(self):
        # Simply checks whether no errors are raised during training
        fit_params = {
            "nb_epoch": 2,
        }

        loss = self.dcmodel.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val, **fit_params)

        self.assertTrue(loss > 0)


    def evaluate(self):
        # Simply checks whether no errors are raised during evaluation

        score_rmse = self.dcmodel.evaluate(self.X_test, self.y_test, metric="RMSE")
        score_mse = self.dcmodel.evaluate(self.X_test, self.y_test, metric="MSE")

        self.assertTrue(score_rmse > 0)
        self.assertTrue(score_mse > 0)


