import io
import logging
import os
import pandas as pd
import rdkit

from mole.utils.mol_utils import parentize_molecules, standardize_molecules


def _get_extension(path_to_file):
    extension = os.path.splitext(path_to_file)[1]
    if extension not in [".csv", ".sdf", ]:
        error_msg = "Currently only supporting CSV and SDF files"
        raise NotImplementedError(error_msg)
    return extension


def read_loaded_files(uploader):
    loaded_names = list(uploader.value.keys())
    file_name = loaded_names[-1]
    file_type = _get_extension(file_name)
    uploaded_file = uploader.value[file_name]["content"]

    if file_type == ".csv":
        df = pd.read_csv(io.BytesIO(uploaded_file))

    elif file_type == ".sdf":
        from rdkit.Chem.PandasTools import LoadSDF
        df = LoadSDF(io.BytesIO(uploaded_file))

    return df


def extract_data(global_data, saved_responses):
    if global_data["use_loaded"]:
        uploader = saved_responses["uploader"]

        try:
            df = read_loaded_files(uploader)
        except Exception as e:
            logging.exception(
                "There was an error when processing the loaded file. Please check whether it is properly formatted and uploaded.")
            raise (e)

        smiles_column_name = saved_responses["smiles_column_name"].value
        assert smiles_column_name in df.columns, f"Provided SMILES column name: \"{smiles_column_name}\" is not present in the dataset."

        smiles = df[smiles_column_name]

    else:
        df = None
        smiles = saved_responses["smiles_input"].value.split("\n")
        smiles = pd.Series(smiles)

    return df, smiles


def process_smiles(smiles, processing_method=None):
    assert processing_method in ["None", "Parentize", "Standardize", "Canonicalize"]

    original_smiles = smiles

    if processing_method == "Parentize":
        new_smiles = parentize_molecules(smiles)
        name = "Parentized SMILES"

    elif processing_method == "Standardize":
        new_smiles = standardize_molecules(smiles)
        name = "Standardized SMILES"

    elif processing_method == "Canonicalize":
        new_smiles = smiles.apply(rdkit.Chem.CanonSmiles)
        name = "Canonicalized SMILES"

    else:
        new_smiles = smiles
        name = "Original SMILES"

    comparison_df = pd.DataFrame({
        "Original SMILES": original_smiles,
        name: new_smiles
    })

    return new_smiles, comparison_df

