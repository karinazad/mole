import logging
import os
from collections import Counter
import pandas as pd
from rdkit import Chem

from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)

DEFAULT_PATH_TO_PKA_TABLE = "data/smarts/latest/smarts_table.csv"
DEFAULT_PATH_TO_PKA_TEST = "data/smarts/latest/smarts_test.csv"

class AnnotatorpKa:
    def __init__(self,
                 path_to_pka_table=None,
                 use_extended_std=True,
                 custom_logger=None):
        """

        Args:
            path_to_pka_table:f
            use_extended_std:f
            custom_logger: f
        """

        self.custom_logger = custom_logger

        if path_to_pka_table is None:
            # Try whether pka table can be found in an expected place
            path_to_pka_table = DEFAULT_PATH_TO_PKA_TABLE

            if not os.path.exists(path_to_pka_table):
                error_msg = f"No path provided and default path to pKa table ({path_to_pka_table}) could not be used."
                logger.exception(error_msg)
                raise ValueError(error_msg)

        pka_table = pd.read_csv(path_to_pka_table)
        pka_table = pka_table.dropna(subset=["group"])
        pka_table = pka_table.sort_values(by="priority", ascending=True)
        self.pka_table = pka_table.copy()

        if use_extended_std:
            print()
            self.pka_table["lower_limit"] = self.pka_table["pka"] - self.pka_table["extended_sd"]
            self.pka_table["upper_limit"] = self.pka_table["pka"] + self.pka_table["extended_sd"]

        else:
            self.pka_table["lower_limit"] = self.pka_table["pka"] - self.pka_table["sd"]
            self.pka_table["upper_limit"] = self.pka_table["pka"] + self.pka_table["sd"]

    def annotate(self, smiles, experimental_pka=None):
        """

        Args:
            smiles: f
            experimental_pka:

        Returns:

        """
        smiles.name = "SMILES"
        smiles = smiles.to_frame()
        smiles["ID"] = smiles.index

        if experimental_pka is not None:
            assert len(experimental_pka) == len(smiles), f"Length mismatch between SMILES and experimental pKa values: " \
                                                         f"n={len(smiles)} for SMILES " \
                                                         f"but n={len(experimental_pka)} for experimental pKa values."

            # Replace zeros with a small value to avoid undesired behavior
            experimental_pka = experimental_pka.replace({0: 1.0e-5})
            smiles["exp_pka"] = experimental_pka
            res = smiles.apply(lambda x: self._get_unique_matches(x.SMILES, x.ID, x.exp_pka), axis=1)

        else:
            res = smiles.apply(lambda x: self._get_unique_matches(x.SMILES, x.ID), axis=1)

        for r in res:
            r["Compound Annotation"] = len(r) * [self._get_pka_class_annotation(r)]

            if experimental_pka is not None:
                r["Experimental pKa in range"] = len(r) * [self._check_in_range(r)]

        df = pd.concat(res.tolist())
        df.set_index("ID", drop=True, inplace=True)
        # df = df.drop(["lower_limit", "upper_limit"], axis=1)
        if experimental_pka is not None:
            df = df.rename(columns={"exp_pka": "Experimental pKa"})
            # df = df.drop(["exp_pka"], axis=1)

        return df

    def _get_matches(self, smi):
        mol = Chem.MolFromSmiles(smi)
        matches = []
        for i in self.pka_table.index:
            group = self.pka_table.loc[i]["smarts"]
            if type(group) == str:
                try:
                    match = mol.GetSubstructMatches(Chem.MolFromSmarts(group))
                    if match:
                        info = self.pka_table.loc[i]
                        info["Matched Atoms"] = match
                        matches.append(info)

                except Exception as e:
                    pass

        if len(matches) == 0:
            info = {col: "no match" for col in self.pka_table.columns}
            info["Matched Atoms"] = "no match"
            matches.append(info)

        return matches

    def _get_unique_matches(self, smi, indx, exp_pka=None):
        try:
            matches = self._get_matches(smi)

            df = pd.DataFrame(matches)
            df = df.explode('Matched Atoms')

            unique_df = []
            groups_to_check = [i for i in range(len(df))]

            while groups_to_check:
                i = groups_to_check.pop(0)
                current = df.iloc[i]
                to_check = [i for i in groups_to_check]

                for j in to_check:
                    compared = df.iloc[j]
                    if any(item in compared["Matched Atoms"] for item in current["Matched Atoms"]):
                        if compared["priority"] < current["priority"]:
                            current = compared
                        else:
                            groups_to_check.remove(j)

                unique_df.append(current)

            unique_df = pd.DataFrame(unique_df)
            unique_df = unique_df.drop_duplicates()
            unique_df.insert(0, "SMILES", smi)
            unique_df.insert(0, "ID", indx)

            if exp_pka:
                unique_df.insert(2, "exp_pka", exp_pka)

            return unique_df

        except Exception as e:
            print(f"\t\tEncountered an error for {indx, smi, e}")
            return None

    def _check_in_range(self, df):
        if "exp_pka" in df.columns:
            try:
                mask_within_range = (df["exp_pka"].astype(float) > df["lower_limit"]) \
                                    & (df["exp_pka"].astype(float) < df["upper_limit"])
                if not any(list(mask_within_range)):
                    return "out of range"
                else:
                    return "in range"
            except:
                return "undetermined"
        else:
            pass

    def _get_pka_class_annotation(self, df):
        classes = set(df["pka_class"].values)
        subclasses = set(df["pka_subclass"].values)

        if "base" in classes and "acid" not in classes:
            return "base"

        elif "base" not in classes and "acid" in classes:
            return "acid"

        elif "base" in classes and "acid" in classes:

            if "strong_acid" in subclasses and "strong_base" not in subclasses:
                return "acid"

            elif "strong_acid" not in subclasses and "strong_base" in subclasses:
                return "base"

            elif ("weak_acid" in subclasses and "weak_base" in subclasses) or \
                    ("strong_acid" in subclasses and "strong_base" in subclasses):

                if "exp_pka" in df.columns:
                    mask_within_range = (df["exp_pka"] > df["lower_limit"]) & (df["exp_pka"] < df["upper_limit"])

                    if not any(list(mask_within_range)):
                        return "zwitterion (out of range)"

                    counter = Counter(df[mask_within_range]["pka_class"])
                    result = counter.most_common(1)[0][0]
                    assert type(result) is str
                    return result

                else:
                    return "zwitterion"

        elif "no match" in classes:
            return "no match"

        else:
            return "neutral"
