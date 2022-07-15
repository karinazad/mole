import logging
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from mole.data.dataloader import DataLoader
from mole.utils.common import setup_logger

RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class TautomerGenerator:
    def __init__(self):
        self.stats = {}

    def tautomerize(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def __call__(self, smiles: list, labels: Optional[list] = None, limit: Optional[int] = None,
                 standardize: bool = False):
        """
        Generates tautomers for a given list of SMILES strings.
        Parameters
        ----------
        smiles: list
            List of of SMILES strings
        labels: Optional(list)
            List of labels or targets associated with SMILES strings.
        limit: Optional(list)
            Limit the number of tautomers per molecule. By default, returns all tautomers.
        Returns
        -------
        tautomer_smiles: list
            Flattened of all generated tautomer SMILES strings.
        augmented_labels: list
            If labels is not None, returns labels that correspond to the original label for each tautomer.
        """
        from functools import reduce

        molecules = [Chem.MolFromSmiles(s) for s in smiles]

        if limit:
            assert limit > 0, "Limit for the number of tautomers must be greater than 0."
            tautomers = [self._get_ordered_tautomers(m, ordered=True)[:limit] for m in tqdm(molecules)]

        else:
            tautomers = [self._get_ordered_tautomers(m, ordered=False) for m in tqdm(molecules)]

        self.stats["tautomers_per_molecule"] = [len(t) for t in tautomers]
        self.stats["num_orig_mol"] = len(smiles)

        if labels is not None:
            assert len(labels) == len(molecules)
            augmented_labels = [[y] * len(t) for y, t in zip(labels, tautomers)]
            augmented_labels = reduce(lambda x, y: x + y, augmented_labels)
            self.stats["original_labels"] = augmented_labels

        tautomers = reduce(lambda x, y: x + y, tautomers)
        self.tautomer_smiles = tautomers

        self.stats["num_taut_mol"] = len(tautomers)

        if standardize:
            dl = DataLoader()
            tautomers = dl.standardize_molecules(tautomers)

        if labels is not None:
            return tautomers, augmented_labels
        else:
            return tautomers

    def show_stats(self):
        num_orig_mol = self.stats["num_orig_mol"]
        num_taut_mol = self.stats["num_taut_mol"]
        print(f"Number of molecules \n\tbefore tautomerization: {num_orig_mol}")
        print(f"\tafter tautomerization: {num_taut_mol}\n")

        nums = pd.Series(np.array(self.stats["tautomers_per_molecule"]).astype(int))
        print(nums.describe())
        # mean_taut = nums.describe()["mean"]
        # std_taut = nums.describe()["std"]
        # print(f"Average number of tautomers per molecule: {round(mean_taut)} +/- {round(std_taut)} (SD)")

        fig = plt.hist(nums, bins=50)
        plt.title("Distribution of tautomers")
        plt.xlabel("Number of generated tautomers")
        plt.ylabel("Number of molecules")
        plt.show()

    def draw_tautomers(self, n: Optional[int] = None):
        """
        Draws the indicated number of generated tautomers.
        Parameters
        ----------
        n: int
            Number of molecules to draw
        """

        assert self.tautomer_smiles is not None
        if n is None:
            n = 25

        svg = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in self.tautomer_smiles[:n]])
        return svg

    @staticmethod
    def _get_ordered_tautomers(m, ordered):
        """
        Returns all tautomers for a single molecule.
        """
        enumerator = rdMolStandardize.TautomerEnumerator()
        canon = enumerator.Canonicalize(m)
        tauts = enumerator.Enumerate(m)
        smis = [Chem.MolToSmiles(x) for x in tauts]

        if not ordered:
            return smis

        else:
            csmi = Chem.MolToSmiles(canon)
            res = [canon]
            stpl = sorted((x, y) for x, y in zip(smis, tauts) if x != csmi)
            smis += [x for x, y in stpl]
            # smis = [Chem.MolToSmiles(x) for x in res]
            return smis
