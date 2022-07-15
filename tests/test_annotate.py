# import subprocess
# import os
# import unittest
# import pandas as pd
# import logging
#
# logging.disable(logging.CRITICAL)
#
# class TestAnnotate(unittest.TestCase):
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
#     def test_pka_groups(self):
#         results_path = "tests/pka_groups.csv"
#         data_path = "tests/test_smiles.csv"
#         task = "pka_groups"
#
#         # Create and temporarily save test SMILES
#         ids = [f"ID0{i + 1}" for i in range(len(self.smiles))]
#         df = pd.DataFrame({"SMILES": self.smiles, "ID": ids})
#         df.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 annotate.py --data-path tests/test_smiles.csv --results-path tests/  --task {task}"],
#             shell=True)
#
#         print(os.listdir("tests"))
#         results = pd.read_csv(results_path)
#
#         os.remove(results_path)
#         os.remove(data_path)
#
#         self.assertEqual(set(results["SMILES"].unique()), set(self.smiles))
#         self.assertEqual(set(results["Compound Annotation"].unique()), {'base', 'no match', 'zwitterion'})
#
#     def test_pka_groups_with_experimental_pka(self):
#         results_path = "tests/pka_groups.csv"
#         data_path = "tests/test_smiles.csv"
#         task = "pka_groups"
#
#         # Create and temporarily save test SMILES
#         smiles = ['CC1=Nc2ccccc2/C1=C/C=C1\\Cc2ccccc2N1C',
#                   'C[C@H]1c2c(F)cncc2N(Cc2ccccc2)C[C@@H](C)N1C(=O)Nc1cccnc1',
#                   'N#Cc1ccc(-c2ccc(C[C@@H](C#N)NC(=O)C3(N)CCCCC3)cc2)cc1',
#                   'OC[C@H]1O[C@H](Oc2ccc(-c3ccc(-c4nnn[nH]4)cc3)cc2)[C@@H](O)[C@@H](O)[C@@H]1O',
#                   'CN1[C@H](C[C@@H](O)c2ccccc2)CCC[C@@H]1CC(=O)c1ccccc1',
#                   'C[C@H]1C[C@@H]2CCC[C@@H]2N1C',
#                   'COc1ccc(S(=O)(=O)NCCN2CCOCC2)c(OC)c1OC',
#                   'O=C(Nc1ccc(C2CCCCC2)cc1)C1=C(O)C(CCc2ccccc2)NC1=O',
#                   'C[Si](C)(C)OC1CN(C2CCCCC2)C1',
#                   'O=C(c1ccc2cc(OCCCN3CCCCC3)ccc2n1)N1CCC1']
#
#         pka = [1.22, 3.8, 7.0999999, 3.7, 8.0, 10.1, 6.1599998, 3.0, 9.0200005, 9.3999996]
#
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         df = pd.DataFrame({"SMILES": smiles, "ID": ids, "Measured_pKa": pka})
#         df.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 annotate.py --data-path tests/test_smiles.csv --results-path tests/  --task {task} --experimental-pka-column Measured_pKa"],
#             shell=True)
#
#         print(os.listdir("tests"))
#         results = pd.read_csv(results_path)
#
#         os.remove(results_path)
#         os.remove(data_path)
#
#         self.assertEqual(set(results["SMILES"].unique()), set(smiles))
#         self.assertTrue("Measured_pKa" in set(results.columns))
#         self.assertTrue("Experimental pKa in range" in set(results.columns))
#
#
#     def test_pka_groups_no_matches_and_compound_annotation(self):
#         results_path = "tests/pka_groups.csv"
#         data_path = "tests/test_smiles.csv"
#         task = "pka_groups"
#
#         smiles = ['Fc1ccc(-c2c[nH]c([C@@H]3CCc4[nH]c5ccccc5c4C3)n2)cc1',
#                   'Fc1ccc(-c2c[nH]c([C@H]3Cc4c([nH]c5ccccc45)CN3)n2)cc1',
#                   'COc1ccc2ncc(C(F)(F)F)c(CCC34CCC(NCc5ccc6c(n5)NC(=O)CO6)(CC3)CO4)c2n1',
#                   'Cc1ccccc1CN1[C@H]2CC[C@@H]1C[C@@H](Oc1cccc(C(N)=O)c1)C2',
#                   'O=[N+]([O-])c1cccc(CNc2cc(C(F)(F)F)cc3ncc(N4CCN(CCO)CC4)cc23)c1',
#                   'CCCC']
#
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         df = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         df.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 annotate.py --data-path tests/test_smiles.csv --results-path tests/  --task {task}"],
#             shell=True)
#
#         results = pd.read_csv(results_path)
#
#         os.remove(results_path)
#         os.remove(data_path)
#
#         assert str(results["Matched Atoms"].iloc[-1]) == "no match"
#
#         # Check a few cases of correct assignment of base, acid, zwitterion, neutral or no match
#         # (these were checked manually)
#         assert str(results["Compound Annotation"].iloc[0]) == "zwitterion"
#         assert str(results["Compound Annotation"].iloc[13]) == "base"
#         assert str(results["Compound Annotation"].iloc[-1]) == "no match"
#
#     def test_issues(self):
#         data_path = "tests/test_smiles.csv"
#         task = "issues"
#         results_path = f"tests/{task}.csv"
#
#         # Create and temporarily save test SMILES
#         smiles = [
#             "COCCCc1cc(CN(C(=O)[C@H]2CNCC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(OCC(C)(C)C(=O)O)c1",
#             "O=c1n(Cc2ccccc2)c2sc3c(c2c2ncnn12)CCN(CC1CCOCC1)C3",
#             "FCFCFC",
#             "NOTAVALIDSMILESSTRING",
#             "CC[C@](O)(c1cn(Cc2ccc3c(-c4ccccc4)c(C(N)=O)sc3c2)nn1)C(F)(F)F",
#             "CCOC(=O)COc1ccc(NC(=O)c2ccc(C(=N)N(C)C)cc2)c(C(=O)Nc2ccc(Cl)cn2)c1",
#             "C",
#             "Cc1ncoc1-c1nnc(SCCCN[C@@H]2CC[C@]3(c4ccc(C(F)(F)F)cc4F)CC23)n1C",
#             "CCP(CC)(CC)=[Au]S[C@H]1O[C@@H](COC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@@H]1OC(C)=O"
#
#         ]
#         ids = [f"ID0{i + 1}" for i in range(len(smiles))]
#         df = pd.DataFrame({"SMILES": smiles, "ID": ids})
#         df.to_csv(data_path)
#
#         # Run predictions
#         subprocess.run(
#             [f"python3 annotate.py --data-path tests/test_smiles.csv --results-path tests/  --task {task}"],
#             shell=True)
#
#         results = pd.read_csv(results_path)
#         os.remove(results_path)
#         os.remove(data_path)
#
#         assert len(results) == len(smiles)
#         assert set(results.columns) == {"ID", "SMILES", "Issues"}
#
#         results = results.set_index("ID")
#         assert results.loc["ID02"]["Issues"] == "()"
#         assert results.loc["ID03"]["Issues"] == "((8, 'Invalid SMILES string: Failed to convert to a molecule.'),)"
#         assert results.loc["ID04"]["Issues"] == "((8, 'Invalid SMILES string: Failed to convert to a molecule.'),)"
#         assert results.loc["ID06"]["Issues"] == "((2, 'InChI: Omitted undefined stereo'),)"
#
#     def test_standardized(self):
#         pass
#
#     def test_custom_columns_and_separator(self):
#         data_path = "tests/test_smiles.csv"
#         task = "issues"
#         results_path = f"tests/{task}.csv"
#
#         # Create and temporarily save test SMILES
#         column1 = [1, 2, 3, 4, 5, 6]
#         column2 = ["A", "B", "C", "D", "E", "F"]
#         ids = [f"ID0{i + 1}" for i in range(len(self.smiles))]
#
#         df = pd.DataFrame({"MySMILES": self.smiles, "MyID": ids, "column1": column1, "column2": column2})
#         df.to_csv(data_path, sep=";")
#
#         # Run predictions
#         subprocess.run(
#             [
#                 f"python3 annotate.py --data-path tests/test_smiles.csv --results-path tests/  --task {task} --sep semicolon --smiles-column MySMILES --id-column MyID"],
#             shell=True)
#
#         results = pd.read_csv(results_path)
#
#         os.remove(results_path)
#         os.remove(data_path)
#
#         assert len(results) == len(self.smiles)
#         assert set(results.columns) == {"MyID", "SMILES", "Issues", "column1", "column2"}
#
#         results = results.set_index("MyID")
#         assert results.loc["ID02"]["Issues"] == "((2, 'InChI: Omitted undefined stereo'),)"
#         assert results.loc["ID03"]["Issues"] == "()"
#
#
# if __name__ == '__main__':
#     unittest.main()
