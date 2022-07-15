import pandas as pd

from src.data_proc.dataloader import DataLoader
from src.utils.common import get_time_stamp


def check_external_data(internal_data, external_data):
    # external_data = pd.read_csv(os.path.join(root_path, "data/logD/external_logD.csv"), sep=";")

    external_data = external_data[external_data["Standard Type"] == "LogD7.4"]

    dataloader = DataLoader()
    stand_external = dataloader.standardize_molecules(external_data.Smiles)
    stand_internal = dataloader.standardize_molecules(internal_data.SMILES)

    external_data["Smiles"] = stand_external
    external_data = external_data.rename(columns={"Smiles": "SMILES"})

    internal_data["SMILES"] = stand_internal

    all_data = pd.concat([internal_data[["SMILES", "logD7.4"]],
                          external_data[["SMILES", "Standard Value"]]]
                         ).drop_duplicates(subset="SMILES")

    print(f"Number of new samples: {len(all_data) - len(internal_data)}")
    num_new = len(all_data) - len(internal_data)

    new_external_data = all_data[-num_new:]
    new_external_data = new_external_data.drop(["logD7.4"], axis=1)
    new_external_data

    stamp = get_time_stamp()

    # new_external_data[["SMILES", "Standard Value"]]
    # .to_csv(os.path.join(root_path, f"data/logD/logD_external_{stamp}.csv"))

    # external_data = external_data.rename(columns={"Standard Value": "logD7.4"})
    # all_data = pd.concat([data[["SMILES", "logD7.4"]], external_data[["SMILES", "logD7.4"]]]).drop_duplicates(
    #     subset="SMILES")
    #
    # all_data = all_data.drop_duplicates(subset="SMILES")
    # all_data

    # # SDF file upload
    # from rdkit.Chem.PandasTools import LoadSDF
    # external_data = LoadSDF(os.path.join(root_path, "data/logP/ncidb.sdf"))
    #
    # # Drop rows witout experimental logP
    # external_data = external_data.dropna(subset=["Experimental logP"])
    # external_data = external_data[["SMILES", "Experimental logP"]]
    #
    # # Standardize SMILES and Replace
    # dataloader = DataLoader()
    # stand_external = dataloader.standardize_molecules(external_data.SMILES)
    # external_data["SMILES"] = stand_external



