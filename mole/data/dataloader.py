import logging
import os
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem

RDLogger.DisableLog('rdApp.*')
from mole.utils.common import check_if_intended
from mole.utils.mol_utils import check_smiles_issues, get_molblock_from_smiles, parentize_molecules
from mole.utils.common import setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger, level="INFO")


class DataLoader:
    def __init__(self, custom_logger=None):
        self.invalid_syntax_indices = []
        self.invalid_chemistry_indices = []
        self.custom_logger = custom_logger

        self.smiles = None
        self.rest = None

    def load_smiles(self,
                    path_to_file,
                    smiles_column=None,
                    id_column=None,
                    parentize=False,
                    check_validity=False,
                    filter_issues=3,
                    log_issues=False,
                    canonicalize=False,
                    separator=None,
                    ):

        file_type = self._get_extension(path_to_file)
        smiles, rest = self._load_smiles(path_to_file, file_type, smiles_column, id_column, separator)
        self._check_if_loaded_valid_smiles(smiles)

        if check_validity:
            issues = check_smiles_issues(smiles)

            # Print existing issue
            issues_ = issues[issues != ()]
            for index, issue in zip(issues_.index, issues_, ):
                for iss in issue:
                    if iss[0] > filter_issues:
                        if log_issues:
                            self._log(f"\n   Index: {index}.\n   SMILES: {smiles.loc[index]}.\n "
                                      f"  Issue(s): {issue}\n\n ")

            max_issue = issues.apply(lambda x: [iss[0] for iss in x])
            max_issue = max_issue.apply(lambda x: max(x) if x else 0)
            smiles = smiles[max_issue < filter_issues]

        if canonicalize:
            smiles = smiles.apply(Chem.CanonSmiles)
        if parentize:
            smiles = parentize_molecules(smiles)

        self.smiles = smiles
        self.rest = rest

        return smiles

    def _load_smiles(self,
                     path_to_file,
                     file_type,
                     smiles_column,
                     id_column,
                     separator=None
                     ):

        if separator is None or separator == "colon" or separator == "," or separator == "c":
            sep = ","
        elif separator == "tab" or separator == "\t" or separator == "t":
            sep = "\t"
        elif separator == "semicolon" or separator == ";" or separator == "s":
            sep = ";"
        else:
            error_msg = f"Unrecognized separator {separator}. Please choose one of the following: " \
                        f"semicolon (s), colon (c), tab (t)"
            logging.exception(error_msg)
            raise ValueError(error_msg)

        if file_type == ".csv":
            df = pd.read_csv(path_to_file, header=0, sep=sep)
            smiles = self._get_smiles_ids_from_df(df=df, smiles_column=smiles_column, id_column=id_column)

            rest = df.copy()
            rest.index = smiles.index
            rest = rest.drop([smiles.name, smiles.index.name, "Unnamed: 0"], axis=1, errors="ignore")

        elif file_type == ".txt":
            df = pd.read_csv(path_to_file, header=None, index_col=None, sep=sep)
            smiles = df.iloc[:, 0]
            smiles.index.name = "ID"
            rest = None

        elif file_type == ".smi":
            df = pd.read_csv(path_to_file, header=None, sep=sep)
            smiles = df.iloc[:, 0]
            rest = None

        elif file_type == ".sdf":
            from rdkit.Chem.PandasTools import LoadSDF
            df = LoadSDF(path_to_file)
            smiles = self._get_smiles_ids_from_df(df=df, smiles_column=smiles_column, id_column=id_column)
            rest = None

        else:
            error_msg = "Cannot load this file. Currently, only txt, csv, smi and sdf are supported."
            self._log(error_msg, level="exception")
            raise NotImplementedError(error_msg)

        return smiles, rest

    def _get_extension(self, path_to_file):
        extension = os.path.splitext(path_to_file)[1]
        if extension not in [".csv", ".txt", ".sdf", ".smi"]:
            error_msg = "Currently only supporting CSV TXT SDF or SMI files"
            self._log(error_msg, level="exception")
            raise NotImplementedError(error_msg)
        return extension

    def _read_sdf_column(self, path, column_name):
        from rdkit.Chem.PandasTools import LoadSDF
        df = LoadSDF(path)

        if column_name not in df.columns:
            error_msg = "Invalid column name provided for SDF."
            self._log(error_msg, level="exception")
            raise ValueError(error_msg)

        column = df[column_name]
        return column

    def _get_smiles_ids_from_df(self, df, smiles_column, id_column):

        # Get SMILES (by default, assumes that SMILES are stored under the column name "SMILES"
        if smiles_column is not None:
            if smiles_column not in set(df.columns):
                error_msg = f"Provided SMILES column name: \"{smiles_column}\" is not present in the dataset."
                self._log(error_msg, level="exception")
                raise ValueError(error_msg)

            smiles = df[smiles_column]

        elif "SMILES" in set(df.columns):
            smiles = df["SMILES"]

        else:
            error_msg = "Failed to retrieve SMILES from the dataset. There was no column SMILES " \
                        "in the provided file. " \
                        "When using a different column name, please provide it as an argument."
            self._log(error_msg, level="exception")
            raise ValueError(error_msg)

        # Get IDs (by default, assumes that IDs are stored under the column name "ID"
        if id_column is not None:
            if id_column not in set(df.columns):
                error_msg = f"Provided ID column name: \"{smiles_column}\" is not present in the dataset."
                self._log(error_msg, level="exception")

            ids = df[id_column]
            smiles.index = ids

        elif "ID" in set(df.columns):
            ids = df["ID"]
            smiles.index = ids

        else:
            msg = "Did not find any ID column in the input file - will continue with new IDs."
            self._log(msg, level="warning")

        return smiles

    def _check_if_loaded_valid_smiles(self, smiles):
        num_smiles = np.minimum(len(smiles), 5)
        valid_smiles = False

        # Try to find at least one valid SMILES
        for i in range(num_smiles):
            smi = smiles.iloc[i]
            molblock = get_molblock_from_smiles(smi)
            if molblock != "":
                valid_smiles = True
                break

        # If all attempts fail, inform the user
        if not valid_smiles:
            self._log("No valid SMILES detected. Informing the user.", level="warning")
            check_if_intended(f"Warning: It seems that there might be a mistake with loaded SMILES strings: "
                              f"\n{smiles.head()} \n Are you sure you want to continue?")

    def _log(self, msg, level="info"):
        if self.custom_logger is not None:
            if level == "info":
                self.custom_logger.info(msg)
            elif level == "warning":
                self.custom_logger.warning(msg)
            elif level == "error":
                self.custom_logger.error(msg)
            elif level == "exception":
                self.custom_logger.exception(msg)
        else:
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "exception":
                logger.exception(msg)
