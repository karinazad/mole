import os
import unittest
import pandas as pd
import numpy as np
from mole.data import DataLoader
from mole.utils.common import silence_stdout
import logging
logging.disable(logging.CRITICAL)

data_path = "../data/test"

class TestDataloader(unittest.TestCase):

    def test_csv(self):
        smiles = [
            "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
            "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
            "CCCC",
            "Cc1ccccc1C=O",
            "CC[C@](O)(c1cn(Cc2ccc3c(-c4ccccc4)c(C(N)=O)sc3c2)nn1)C(F)(F)F",
            "C"
        ]
        df = pd.DataFrame({"SMILES": smiles})
        df.to_csv("test_smiles.csv")
        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.csv")
        os.remove("test_smiles.csv")

        self.assertEqual(len(smiles), len(loaded_smiles))
        self.assertTrue(all([x == y for x,y in zip(smiles, loaded_smiles.tolist())]))

    def test_ids_csv(self):
        smiles = [
            "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
            "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
            "CCCC",
        ]
        ids = [
            "IDO1",
            "IDO2",
            "IDO3",
        ]
        df = pd.DataFrame({"SMILES": smiles, "ID": ids})
        df.to_csv("test_smiles.csv")

        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.csv")

        self.assertEqual(ids, list(loaded_smiles.index))
        self.assertEqual(smiles, list(loaded_smiles.values))

        os.remove("test_smiles.csv")

    def test_custom_separators_and_names_csv(self):
        smiles = [
            "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
            "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
            "CCCC",
        ]
        ids = [
            "IDO1",
            "IDO2",
            "IDO3",
        ]
        # Semi-colon separator (semicolon)
        df = pd.DataFrame({"MySMILES": smiles, "MyIDs": ids})
        df.to_csv("test_smiles.csv", sep=";")

        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.csv", separator="semicolon", smiles_column="MySMILES",
                                               id_column="MyIDs")

        self.assertEqual(list(loaded_smiles.index), ids)
        self.assertTrue(all([x == y for x, y in zip(smiles, loaded_smiles.tolist())]))

        # Semi-colon separator abbreviation (s)
        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.csv", separator="s", smiles_column="MySMILES",
                                               id_column="MyIDs")

        self.assertEqual(list(loaded_smiles.index), ids)
        self.assertTrue(all([x == y for x, y in zip(smiles, loaded_smiles.tolist())]))

        # Tab separator (tab)
        df = pd.DataFrame({"MySMILES": smiles, "MyIDs": ids})
        df.to_csv("test_smiles.csv", sep="\t")

        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.csv", separator="tab", smiles_column="MySMILES",
                                               id_column="MyIDs")

        self.assertEqual(list(loaded_smiles.index), ids)
        self.assertTrue(all([x == y for x, y in zip(smiles, loaded_smiles.tolist())]))

        os.remove("test_smiles.csv")

    def test_invalid_smiles_loading_warning(self):
        not_smiles = [
            "These",
            "Are",
            "Not SMILES",
        ]
        df = pd.DataFrame({"SMILES": not_smiles})
        df.to_csv("test_smiles.txt", header=None, index=None)

        dataloader = DataLoader()

        # Loading these SMILES should result in an error in pytest (As it asks a user for additional input)
        try:
            dataloader.load_smiles("test_smiles.txt")
            raise ValueError("Expected OSError.")
        except:
            pass
        os.remove("test_smiles.txt")


    def test_txt(self):
        smiles = [
            "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
            "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
            "CCCC",
            "Cc1ccccc1C=O",
            "CC[C@](O)(c1cn(Cc2ccc3c(-c4ccccc4)c(C(N)=O)sc3c2)nn1)C(F)(F)F",
            "C"
        ]
        smiles = pd.DataFrame({"SMILES": smiles}, index=np.arange(len(smiles)))
        smiles.to_csv("test_smiles.txt", header=None, index=None)

        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.txt")
        os.remove("test_smiles.txt")

        self.assertEqual(len(smiles), len(loaded_smiles))
        self.assertTrue(all([x == y for x, y in zip(smiles["SMILES"].tolist(), loaded_smiles.tolist())]))


    def test_sdf(self):
        # SDF test file must be saved locally, if not skip this test
        try:
            path_to_file = os.path.join(data_path, 'smiles.sdf')
            dataloader = DataLoader()
            smiles = dataloader.load_smiles(path_to_file)
        except FileNotFoundError:
            pass

    def test_valid_and_invalid_smiles(self):
        smiles = [
            "CGHNB",
            "CCC1ccC",
            "COc1ccc([C@@H](O)C[C@H]2c3cc(OC)c(OC)cc3CCN2C)cc1",
            "CFFF"
        ]
        smiles = pd.DataFrame({"SMILES": smiles} )#, index=np.arange(len(smiles)))
        smiles.to_csv("test_smiles.csv")

        dataloader = DataLoader()

        with silence_stdout():
            loaded_smiles = dataloader.load_smiles("test_smiles.csv", check_validity=True)

        self.assertEqual(len(loaded_smiles), 1)
        self.assertEqual(loaded_smiles.values[0], "COc1ccc([C@@H](O)C[C@H]2c3cc(OC)c(OC)cc3CCN2C)cc1")

        os.remove("test_smiles.csv")

    def test_rest(self):
        smiles = [
            "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
            "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
            "CCCC",
        ]
        ids = [
            "IDO1",
            "IDO2",
            "IDO3",
        ]
        column1 = [
            "value1A",
            "value1B",
            "value1C"
        ]
        column2 = [
            "value2A",
            "value2B",
            "value2C"
        ]
        df = pd.DataFrame({"SMILES": smiles, "MyID": ids, "column1": column1, "column2": column2})
        df.to_csv("test_smiles.csv")

        dataloader = DataLoader()
        loaded_smiles = dataloader.load_smiles("test_smiles.csv", id_column="MyID")

        self.assertEqual(ids, list(loaded_smiles.index))
        self.assertEqual(smiles, list(loaded_smiles.values))

        combined = pd.merge(loaded_smiles, dataloader.rest, left_index=True, right_index=True)

        self.assertEqual(set(df.columns), set(list(combined.columns) + [combined.index.name]))

        os.remove("test_smiles.csv")


if __name__ == '__main__':
    unittest.main()


