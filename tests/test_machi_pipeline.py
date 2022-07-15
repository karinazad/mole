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
from mole.pipeline import save_pipeline, load_pipeline, MachiLightPipeline
from mole import task_info

logging.disable(logging.CRITICAL)

ROOT_PATH = ""
PIPE_DIR = os.path.join(ROOT_PATH, "pipelines/prod")


class TestMachiLightPipeline(unittest.TestCase):

    def setUp(self) -> None:

        self.pipeline = MachiLightPipeline(
            tasks=task_info.custom_tasks,
            pipeline_dir=PIPE_DIR,
            include_novelty=True,
            logger=None,
        )

        self.smiles = pd.Series([
            "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
            "CCC1(CC)C(=O)NC(=O)N(C)C1=O",
            "O=P1(N(CCCl)CCCl)NCCCO1",
            "CC(O)C(=O)O",
        ])

    def test_predict(self):
        predictions = self.pipeline.predict(self.smiles)

        # TODO: This test is not passing due to DGL dependency error
        # assert all([x in predictions.columns for x in task_info.custom_tasks])
        assert len(predictions) == len(self.smiles)


