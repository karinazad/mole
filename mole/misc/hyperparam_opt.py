import argparse
import wandb
import deepchem as dc
from deepchem.models import AttentiveFPModel
import numpy as np
from sklearn.metrics import mean_squared_error
from mole.data import DataLoader
from mole.featurizers import AttentiveFPFeaturizer
from mole.utils.train_utils import early_stop



def hyperparam_opt(args):
    data_path = args.data_path
    val_data_path = args.separate_val_path
    test_data_path = args.separate_test_path

    featurizer = AttentiveFPFeaturizer()
    train_dataset, val_dataset, _ = get_dc_datasets(data_path, val_data_path, test_data_path, featurizer=featurizer)
    sweep_config = {
        "name": "sweep",
        "method": "random",
        "metric": {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10
        },
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.1},
            "batch_size": {"values": [8, 16, 32, 64, 128]},
            "epochs": {"value": 50},
        }
    }
    sweep_id = wandb.sweep(sweep_config)

    def train():
        with wandb.init(project=args.task_name.split("_")[0], entity="themamaai_karina") as run:
            prev_val_loss, n_no_improvement = float("inf"), 0
            config = wandb.config
            model = AttentiveFPModel(mode="regression",
                                     n_tasks=1,
                                     batch_size=config.batch_size,
                                     learning_rate=config.learning_rate)

            for epoch in range(config.epochs):
                loss = model.fit(train_dataset, nb_epoch=1)

                y_pred = np.array(model.predict(val_dataset)).flatten()
                y_true = np.array(val_dataset.y).flatten()
                val_loss = mean_squared_error(y_true, y_pred, squared=False)

                wandb.log({"loss": loss, "epoch": epoch})
                wandb.log({"val_loss": val_loss, "epoch": epoch})

                prev_val_loss, n_no_improvement = early_stop(loss, prev_val_loss, n_no_improvement)

                if n_no_improvement >= config.early_stopping_patience:
                    wandb.log({"final_number_of_epochs": epoch})
                    break

                run.finish()

    wandb.agent(sweep_id, function=train, count=1)
    wandb.finish()


def get_dc_datasets(train_data_path, val_data_path, test_data_path, featurizer):
    print("\n\tLoading data...")
    dataloader = DataLoader()
    smiles = dataloader.load_smiles(train_data_path)
    y = dataloader.rest

    dataloader = DataLoader()
    smiles_val = dataloader.load_smiles(val_data_path)
    y_val = dataloader.rest

    dataloader = DataLoader()
    smiles_test = dataloader.load_smiles(test_data_path)
    y_test = dataloader.rest

    print("\n\tFeaturizing data...")
    X = featurizer.transform(smiles)
    train_dataset = dc.data.NumpyDataset(X=X, y=y)

    X_val = featurizer.transform(smiles_val)
    val_dataset = dc.data.NumpyDataset(X=X_val, y=y_val)

    X_test = featurizer.transform(smiles_test)
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="AttentiveFP",
        choices=["Chemprop", "AttentiveFP"],
        required=False,
        help="A path to a csv file with SMILES in the first column and target in the second. Must contain a header.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="A path to a csv file with SMILES in the first column and target in the second. Must contain a header.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=False,
        default="regression",
        choices=["classification", "regression"],
        help="Either “classification” or “regression” depending on the type of the dataset"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="A directory where model checkpoints will be saved.",
    )
    parser.add_argument(
        "--separate-test-path",
        type=str,
        default=None,
        help="A path to a csv file with TEST SMILES in the first column and target in the second. Must contain a header"
             "Hint: If you have separate data files you would like to use as the validation or test set,"
             "you can specify them with --separate_val_path <val_path> and/or --separate_test_path <test_path>..",
    )
    parser.add_argument(
        "--separate-val-path",
        type=str,
        default=None,
        help="A path to a csv file with VAL SMILES in the first column and target in the second. Must contain a header"
             "Hint: If you have separate data files you would like to use as the validation or test set,"
             "you can specify them with --separate_val_path <val_path> and/or --separate_test_path <test_path>..",
    )

    args = parser.parse_args()

    if args.model == "AttentiveFP":
        hyperparam_opt(args)
    else:
        raise NotImplementedError
