import os
import shutil

import pandas as pd
import unittest
import numpy as np

import logging

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from mole.featurizers import RDKitFeaturizer
from mole.models import SklearnModel
from mole.pipeline import save_pipeline, load_pipeline

logging.disable(logging.CRITICAL)

ROOT_PATH = ""
TEST_PATH = os.path.dirname(os.path.realpath(__file__))
PIPE_SAVE_DIR = os.path.join(TEST_PATH, "temp/test_pipe")


class TestSklearnPipeline(unittest.TestCase):

    def setUp(self) -> None:
        # Prepare Data
        # Include also invalid SMILES to check whether it is successfully removed
        self.smiles = pd.Series([
            "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
            "CCC1(CC)C(=O)NC(=O)N(C)C1=O",
            "O=P1(N(CCCl)CCCl)NCCCO1",
            "CC(O)C(=O)O",
            # "DEFINITELY NOT SMILES STRING",
            # "",
            # "XAVIER"
        ])
        self.y = pd.Series([0, 1, 1, 0,])# 0.5, 0.0, -0.1])

        self.assertTrue(len(self.smiles) == len(self.y))

        self.featurizer = RDKitFeaturizer()
        self.sklearn_model = SklearnModel(
            model=RandomForestClassifier(),
            mode="classification",
            num_classes=2
        )
        self.pipeline = Pipeline(steps=[
            ('featurizer', self.featurizer),
            ('model', self.sklearn_model)
        ])

        self.pipeline.fit(self.smiles, self.y, featurizer__num_samples=len(self.smiles))

        os.makedirs(PIPE_SAVE_DIR, exist_ok=True)
        save_pipeline(self.pipeline, PIPE_SAVE_DIR)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(PIPE_SAVE_DIR)

    def test_pipeline_inference(self):
        # Create the pipeline
        preds = self.pipeline.predict(self.smiles)

        # Compare with manual featurization and inference with a normal model
        X, y = self.featurizer.transform(self.smiles), self.y
        preds_model = self.sklearn_model.predict(X)
        self.assertTrue(np.allclose(preds, preds_model))


    def test_pipeline_loading(self):
        # Create the pipeline
        loaded_pipeline = load_pipeline(PIPE_SAVE_DIR)

        preds = self.pipeline.predict(self.smiles)
        preds_from_loaded = loaded_pipeline.predict(self.smiles)

        self.assertTrue(np.allclose(preds, preds_from_loaded))

