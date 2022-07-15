# import sys
# import unittest
#
# import pandas as pd
# import deepchem as dc
# import numpy as np
#
#
# from mole.data.splitter import HierarchicalSplitter
#
#
# class TestHierarchicalSplitter(unittest.TestCase):
#
#     def setUp(self) -> None:
#         task="logP"
#         data_folder="../data"
#         df=pd.read_csv(f"{data_folder}/{task}/{task}.csv").dropna()
#         self.dataset = dc.data.DiskDataset.from_numpy(X=np.array(df["ID"]),y=np.array(df[task]).reshape(-1,1), ids=np.array(df["SMILES"]),tasks=[task])
#         self.splitter=HierarchicalSplitter()
#
#     def test_evaluate_threshold(self,sim_th=0.34):
#         # Chose a similarity threshold and evaluate it
#         train,val,test=self.splitter.split(self.dataset,sim_th=sim_th,evaluate=True)
#
#     def test_splitter(self,frac_train=0.8,frac_valid=0.1,frac_test=0.1):
#         # Split the dataset
#         train,val,test=self.splitter.split(self.dataset,frac_train=frac_train,frac_valid=frac_valid,frac_test=frac_test
#                                             ,sim_th=0.246,evaluate=False)
#         print("Training Dataset : ", train.shape)
#         print("Validation Dataset : ", val.shape)
#         print("Test Dataset : ", test.shape)
#
#         train.to_csv("logP_train.csv")
#         val.to_csv("logP_val.csv")
#         test.to_csv("logP_test.csv")
#
#
#
# if __name__ == '__main__':
#     unittest.main()