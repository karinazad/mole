import unittest
import math
import pandas as pd
import numpy as np
import logging

import deepchem as dc
from mole.featurizers import AttentiveFPFeaturizer

logging.disable(logging.CRITICAL)


class TestAttentiveFPFeaturizers(unittest.TestCase):

    def test1(self):
        smiles = [
            "O=C1CCCN1C1CCN(CCc2ccc(Oc3nc4ccccc4s3)cc2)CC1",
            "COCCOc1ccc2ncc(F)c(CCC34CCC(N=CC5=CC=C6OC=C(O)NC6N5)(CC3)CO4)c2n1",
            "CCCC",
            "Cc1ccccc1C=O",
            "CC[C@](O)(c1cn(Cc2ccc3c(-c4ccccc4)c(C(N)=O)sc3c2)nn1)C(F)(F)F",
            "C"
        ]
        smiles = pd.Series(smiles, index=np.arange(len(smiles)))
        featurizer = AttentiveFPFeaturizer()

        X = featurizer.transform(smiles)
        self.assertEqual(6, len(X))

    def test_featurizable_and_unfeaturizable_smiles(self):
        """
        Should leave an empty array for SMILES that failed to be featurized.
        """
        smiles = [
            "COc1ccc([C@@H](O)C[C@H]2c3cc(OC)c(OC)cc3CCN2C)cc1",
            "[Cr+3]",
            "[Pb + 2]",
            "[Hg + 2]",
            "Fc1ccc(-c2c[nH]c([C@@H]3CCc4[nH]c5ccccc5c4C3)n2)cc1",
            "[Co + 2]",
            "[Fe + 3]",
            "CC1=CCC(C(C)(C)O)CC1",
            "CC[N+](CC)(CC(=O)Nc1c(C)cccc1C)Cc1ccccc1"
        ]
        indices_of_unfeaturizable_smiles = [1, 2, 3, 5, 6]
        indices_of_featurizable_smiles = [0, 4, 7, 8 ]

        smiles = pd.Series(smiles, index=np.arange(len(smiles)))
        featurizer = AttentiveFPFeaturizer()

        X = featurizer.transform(smiles)
        self.assertEqual(9, len(X))
        self.assertTrue(all([not X.iloc[i] for i in indices_of_unfeaturizable_smiles]))
        self.assertTrue(all([type(X.iloc[i]) == dc.feat.graph_data.GraphData
                             for i in indices_of_featurizable_smiles]))

    def test_unfeaturizable_smiles(self):
        """
        Should leave an empty array for SMILES that failed to be featurized.
        """
        smiles = [
            "[Cr+3]",
            "[Pb + 2]",
            "[Hg + 2]",
            "[Co + 2]",
            "[Fe + 3]",
            "NOTSMILES",
            ""

        ]
        smiles = pd.Series(smiles, index=np.arange(len(smiles)))
        featurizer = AttentiveFPFeaturizer()
        X = featurizer.transform(smiles)

        self.assertTrue(all([not x for x in X]))


if __name__ == '__main__':
    unittest.main()
