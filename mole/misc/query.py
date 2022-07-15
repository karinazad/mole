import ast
from typing import Optional

import pandas as pd
from chembl_webresource_client.new_client import new_client


class ChEMBLQuery:
    """
    A query to obtain bioactivity and compound data for a given target
    """

    def __init__(self, verbose: int = 2):
        """
        Parameters
        ----------
        verbose: int
        Indicates how often to update about data collection. For a single task, select 0 or 1. For batch processing,
        2 is recommended.
        """
        self.targets_api = new_client.target
        self.compounds_api = new_client.molecule
        self.bioactivities_api = new_client.activity

        self.compounds_df = None
        self.bioactivities_df = None

        self.verbose = verbose

    def get_bioactivities_and_compounds(self,
                                        target_chembl_id: str,
                                        assay_measurement:str = "IC50",
                                        standard_unit: str = "nM"):
        """
        Returns a dataset of SMILES and measurements that were retrieved for a given target.
        ----------
        target_chembl_id: str
            Unique ChEMBL identifier of a target. For example, CHEMBL1876
        assay_measurement: str
            Select one of "IC50" or "Ki". Other measurements could be available based on data for each target.
        standard_unit: str
            Units of measurement to be kept. For example, only "nM" measurements or "%".
        Returns
        -------
        pd.DataFrame
            Dataframe with columns: molecule_chembl_id,  standard_value, standard_unit, canonical_smiles.
            "standard_value" correspond to the target measurement value.
        """
        print(f"ChEMBL Target ID: {target_chembl_id}")

        bioactivities_df = self.get_bioactivities_df(target_chembl_id=target_chembl_id,
                                                     assay_measurement=assay_measurement,
                                                     standard_unit=standard_unit)

        compounds_df = self.get_compounds_df(molecule_chembl_id=list(bioactivities_df["molecule_chembl_id"]))

        merged_df = pd.merge(
            bioactivities_df[["molecule_chembl_id", "standard_value", "standard_units"]],
            compounds_df,
            on="molecule_chembl_id",
        )

        merged_df.reset_index(drop=True, inplace=True)
        if self.verbose > 0:
            print(f"\tFinal number of samples:  {merged_df.shape[0]}\n\n")

        return merged_df

    def get_targets(self, target_chembl_id):
        raise NotImplemented

    def get_bioactivities_df(self, target_chembl_id, assay_measurement="IC50", standard_unit="nM"):

        bioactivities = self.bioactivities_api.filter(
            target_chembl_id=target_chembl_id, type=assay_measurement, relation="=", assay_type="B",
        ).only(
            "assay_type",
            "molecule_chembl_id",
            "type",
            "standard_units",
            "standard_value",
            "target_chembl_id",
        )

        if len(bioactivities) == 0:
            print("\tNo bioactivity records found \n\n")
            raise UserWarning

        bioactivities_df = pd.DataFrame.from_records(bioactivities)
        bioactivities_df = self._process_queried_dataset(df=bioactivities_df, standard_unit=standard_unit)
        if self.verbose > 0:
            print(f"\tInitial number of measurements: {len(bioactivities)}")
            print(f"\tNumber of clean bioactivity measurements: {len(bioactivities_df)}")

        self.bioactivities_df = bioactivities_df

        return bioactivities_df

    def get_compounds_df(self, molecule_chembl_id: list):

        compounds = self.compounds_api.filter(molecule_chembl_id__in=molecule_chembl_id
                                              ).only("molecule_chembl_id", "molecule_structures")

        if self.verbose > 0:
            if len(compounds) == 0:
                print("\tNo compound records found.")
                return None

        compounds_df = pd.DataFrame.from_records(compounds)
        compounds_df = self._process_queried_dataset(df=compounds_df)
        compounds_df = self._replace_smiles_column(compounds_df)

        if self.verbose > 0:
            print(f"\tNumber of compounds: {len(compounds_df)}")

        self.compounds_df = compounds_df

        return compounds_df

    def _process_queried_dataset(
            self,
            df: pd.DataFrame,
            drop_na: bool = True,
            drop_duplicates: bool = True,
            reset_index=True,
            standard_unit: Optional[str] = None,
            columns_to_float: Optional[list] = None,
    ):
        """
        1. Convert standard_value's datatype from object to float
        2. Delete entries with missing values
        3. Keep only entries with standard_unit == nM
        4. Delete duplicate molecules
        5. Reset DataFrame index
        """
        n_samples = len(df)
        if self.verbose == 2:
            print(f"\tInitial number of samples: {n_samples} \n")

        # Convert standard_value's datatype from object to float
        if columns_to_float is not None:
            df = df.astype({column: "float64" for column in columns_to_float})

        # Delete entries with missing values
        if drop_na:
            df.dropna(axis=0, how="any", inplace=True)
            if self.verbose == 2:
                print(f"\t\t Entries with NaNs: {n_samples - len(df)}."
                      f"\n\t\t New number of samples: {len(df)} \n")
            n_samples = len(df)

        # 3 Keep only entries with standard_unit
        if standard_unit != "all" and standard_unit is not None:
            assert "standard_units" in df.columns

            df = df[df["standard_units"] == standard_unit]

            if self.verbose == 2:
                print(f"\t\t Entries with non-standard units: {n_samples - len(df)}"
                      f"\n\t\t New number of samples: {len(df)}\n")

            n_samples = len(df)

        # Delete duplicate molecules
        if drop_duplicates:
            df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)

            if self.verbose == 2:
                print(f"\t\t Entries duplicates: {n_samples - len(df)}"
                      f"\n\t\t New number of samples: {len(df)}\n")
            n_samples = len(df)

        # Reset DataFrame index
        if reset_index:
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def _replace_smiles_column(df):
        """
        Replaces dictionary structure returned by ChEMBL queries with a single canonical SMILE.
        """

        assert "molecule_structures" in df.columns

        # Convert strings to dictionaries if needed
        if type(df.iloc[0].molecule_structures) is not dict:
            print(type(df.iloc[0].molecule_structures))
            df.molecule_structures = df.molecule_structures.apply(lambda x: ast.literal_eval(x))

        # Keep only SMILES representation
        df["canonical_smiles"] = [
            compounds["molecule_structures"]["canonical_smiles"] for _, compounds in df.iterrows()
        ]

        df.drop("molecule_structures", axis=1, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)

        return df