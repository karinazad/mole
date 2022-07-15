import json
import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mole import task_info
from mole.featurizers import RDKitFeaturizer, AttentiveFPFeaturizer
from mole.models import SklearnModel, DeepChemModel, ChempropModel

try:
    from mole.models import OODDetector
    ood_detection_available = True
except ModuleNotFoundError:
    ood_detection_available = False


class MachiLightPipeline:
    def __init__(self,
                 pipeline_dir: str = None,
                 tasks: list = None,
                 include_novelty: bool = False,
                 logger=None,
                 ):

        # Use logger if provided
        if logger is not None:
            self.logger = logger
            self.logger.debug("\n\n\n Starting MACHI Prediction Pipeline")
        else:
            self.logger = None

        # Checks whether OOD detection is required and whether modules were successfully imported
        if not ood_detection_available and include_novelty:
            logger.warning("OOD Detection is not available")
            self.include_novelty = False
        else:
            self.include_novelty = include_novelty

        # Select tasks to predict for
        self.available_tasks = task_info.custom_tasks

        # If "all" was chosen, use all available tasks
        if tasks == "all" or tasks == ["all"] or tasks is None:
            self.requested_custom_tasks = self.available_tasks

        else:
            # Confirm that tasks is a list of strings
            try:
                assert type(tasks) == list
                assert all([type(s) == str for s in tasks])
            except Exception as e:
                if self.logger:
                    self.logger.error("Exception occured.")
                raise e

            # Filter available tasks
            self.requested_custom_tasks = list(set(self.available_tasks) & set(tasks))

            if self.requested_custom_tasks != len(tasks):
                missing_tasks = list(set(self.available_tasks) & set(tasks))

        # Directory where to find saved pipelines
        self.pipeline_dir = pipeline_dir

        if self.logger:
            self.logger.debug(f"Custom tasks:  {str(self.requested_custom_tasks)}")
            self.logger.debug(f"Pipeline Directory: {str(self.pipeline_dir)}")

    def predict(self, smiles):
        self.predictions = [pd.DataFrame({"SMILES": smiles})]

        self.predict_custom_tasks(smiles)

        if not self.predictions:
            error_msg = "Failed to predict for all properties."
            if self.logger:
                self.logger.error(error_msg)
            raise UserWarning(error_msg)

        if len(self.predictions) > 1:
            result = pd.concat(self.predictions, axis=1)
        else:
            result = self.predictions[0]

        if self.logger:
            self.logger.debug(f"Number of predicted outputs (including novelty predictions): {len(self.predictions)}")
            self.logger.debug("\nFinished MACHI Prediction Pipeline \n\n\n ")

        return result

    def predict_custom_tasks(self, smiles):

        if not len(self.requested_custom_tasks):
            return None

        for task in self.requested_custom_tasks:
            if self.logger:
                self.logger.debug(f"Predicting for {task}...")
            print("\t\t", task)

            try:
                pipeline = load_pipeline(os.path.join(self.pipeline_dir, task))
                pred = pipeline.predict(smiles)
                pred = pred.apply(lambda x: np.round(x, 3) if x is not np.nan else x)

                task_name = task_info.custom_tasks_annot[task]
                pred_df = pd.DataFrame({task_name: pred})
                self.predictions.append(pred_df)

                if self.logger:
                    self.logger.debug(f"\t\t...prediction successfully finished.")

            except Exception as e:
                self._print_prediction_error(task, e)

            if self.include_novelty:
                try:
                    ood_pipeline = load_pipeline(os.path.join(self.pipeline_dir, task, "OOD"))
                    ood_pred = ood_pipeline.predict(smiles)
                    ood_pred = ood_pred.apply(lambda x: np.round(x, 3) if x is not np.nan else x)

                    task_name = f"{task} (novelty score)"
                    ood_pred_df = pd.DataFrame({task_name: ood_pred})
                    self.predictions.append(ood_pred_df)

                    if self.logger:
                        self.logger.debug(f"\t\t...prediction of novelty successfully finished.")

                except Exception as e:
                    self._print_prediction_error(task, e, novelty=True)

    def _print_prediction_error(self, task, e, novelty=False):
        if novelty:
            error_msg = f"\t\t...failed to predict novelty scores for {task}."
        else:
            error_msg = f"\t\t...failed to predict {task}."
        if self.logger:
            self.logger.warning(error_msg)
        print(error_msg, e)


def save_pipeline(pipeline, save_dir):
    model = pipeline["model"]
    featurizer = pipeline["featurizer"]

    model.save(os.path.join(save_dir, "model"))
    featurizer.save(os.path.join(save_dir, "featurizer"))

    pipeline_steps = {
        "model_class": type(model).__name__,
        "featurizer_class": type(featurizer).__name__,
    }

    with open(os.path.join(save_dir, "pipeline_steps.json"), 'w') as fp:
        json.dump(pipeline_steps, fp, indent=4)


def load_pipeline(save_dir):
    with open(os.path.join(save_dir, "pipeline_steps.json")) as f:
        pipeline_steps = json.load(f)

    # Load a model
    model_class = pipeline_steps.get("model_class")

    if model_class == "SklearnModel":
        model = SklearnModel()
    elif model_class == "DeepChemModel":
        model = DeepChemModel()
    elif model_class == "ChempropModel":
        model = ChempropModel()
    elif model_class == "OODDetector":
        model = OODDetector()
    else:
        error_msg = f"Load Pipeline: Unknown model class encountered: {model_class}"
        raise NotImplementedError(error_msg)

    model.restore(os.path.join(save_dir, "model"))

    # Load a featurizer
    featurizer_class = pipeline_steps.get("featurizer_class", None)
    if featurizer_class == "RDKitFeaturizer":
        featurizer = RDKitFeaturizer()
    elif featurizer_class == "AttentiveFPFeaturizer":
        featurizer = AttentiveFPFeaturizer()

    # ChempropModel does not need a featurizer
    elif featurizer_class is None and model_class == "ChempropModel":
        pipeline = Pipeline(steps=[
            ('model', model)
        ])
        return pipeline

    else:
        error_msg = f"Load Pipeline: Unknown featurizer class encountered: {featurizer_class}"
        raise NotImplementedError(error_msg)

    featurizer.restore(os.path.join(save_dir, "featurizer"))

    pipeline = Pipeline(steps=[
        ('featurizer', featurizer),
        ('model', model)
    ])

    return pipeline
