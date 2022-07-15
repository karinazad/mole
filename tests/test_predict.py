# import subprocess
# import os
# import unittest
# import pandas as pd
#
# from machi.task_info import custom_tasks, custom_tasks_annot, fpadmet_num_task_map, fpadmet_tasks
#
# import logging
# logging.disable(logging.CRITICAL)
#
# class TestPredict(unittest.TestCase):
#
#     def setUp(self) -> None:
#         self.smiles = [
#             "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
#             "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
#             "CCCC",
#             "Cc1ccccc1C=O",
#             "CC[C@](O)(c1cn(Cc2ccc3c(-c4ccccc4)c(C(N)=O)sc3c2)nn1)C(F)(F)F",
#             "C"
#         ]
#
#     def test_custom_tasks(self):
#
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         tasks = " ".join(custom_tasks)
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         smiles = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         smiles.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks}"],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#         for task_name in set(custom_tasks_annot.values()):
#             assert task_name in set(predictions.columns)
#         assert len(predictions) == len(smiles)
#
#     def test_FPADMET_tasks_by_code(self):
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         task_codes = list(fpadmet_num_task_map.keys())[:5]
#         tasks = " ".join(task_codes)
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         smiles = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         smiles.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks}"],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#         for task_code in task_codes:
#             task_name = fpadmet_num_task_map[task_code]
#             assert task_name in set(predictions.columns)
#
#         assert len(predictions) == len(smiles)
#
#     def test_FPADMET_tasks_by_name(self):
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         task_names = list(fpadmet_num_task_map.values())[5:10]
#         tasks = " ".join(task_names)
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         smiles = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         smiles.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks}"],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#         # Check that all predictions work as intended
#         for task_name in task_names:
#             assert task_name in set(predictions.columns)
#
#         # Check that all input SMILES are present and their IDs are preserved
#         assert len(predictions) == len(smiles)
#
#     def test_FPADMET_tasks_all(self):
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         tasks = "all"
#         task_names = [task for task in fpadmet_tasks.keys() if not task.startswith("FP")]
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         smiles = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         smiles.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks}"],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#
#         # Check that all predictions work as intended
#         for task_name in task_names:
#             assert task_name in set(predictions.columns)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#
#     def test_novelty_on_custom_tasks(self):
#
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         tasks = " ".join(custom_tasks)
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         smiles = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         smiles.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [
#                 f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks} --novelty True"],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#         # Check that normal predictions are still fine
#         for task_name in set(custom_tasks_annot.values()):
#             assert task_name in set(predictions.columns)
#
#         # Check that OOD predictions are present
#         for task_name in set(custom_tasks_annot.keys()):
#             ood_task_name = f"{task_name} (novelty score)"
#             assert ood_task_name in set(predictions.columns)
#             # assert all(predictions[ood_task_name] <= 1)
#             # assert all(predictions[ood_task_name] >= 0)
#
#         # Check that all input SMILES are present and their IDs are preserved
#         assert len(predictions) == len(smiles)
#
#     def test_additional_columns(self):
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         tasks = " ".join(custom_tasks)
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         column1 = [1, 2, 3, 4, 5, 6]
#         column2 = ["A", "B", "C", "D", "E", "F"]
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#
#         df = pd.DataFrame({"SMILES": smiles, "ID": ids, "column1": column1, "column2": column2})
#         df.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks}"],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#         print(predictions)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#         assert "column1" in list(predictions.columns)
#         assert "column2" in list(predictions.columns)
#         assert len(predictions) == len(smiles)
#         for task_name in set(custom_tasks_annot.values()):
#             assert task_name in set(predictions.columns)
#
#     def test_custom_column_name_and_separators(self):
#
#         pred_path = "tests/predictions.csv"
#         data_path = "tests/test_smiles.csv"
#         tasks = " ".join(custom_tasks)
#
#         # Create and temporarily save test SMILES
#         smiles = self.smiles
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#
#         df = pd.DataFrame({"MySMILES": smiles, "MyID": ids})
#         df.to_csv(data_path, sep=";")
#
#         # Run predictions
#         subprocess.run(
#             [
#                 f"python3 predict.py --data-path tests/test_smiles.csv --results-path tests/  --tasks {tasks} --sep s --smiles-column MySMILES --id-column MyID"
#             ],
#             shell=True)
#
#         predictions = pd.read_csv(pred_path)
#
#         os.remove(pred_path)
#         os.remove(data_path)
#
#         self.assertEqual(len(predictions), len(smiles))
#
#         for task_name in set(custom_tasks_annot.values()):
#             assert task_name in set(predictions.columns)
#
#
# if __name__ == '__main__':
#     unittest.main()
