from typing import Union, Optional
import logging
import os
import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mole.data.tautomers import TautomerGenerator
from mole.utils.common import get_time_stamp, makedirs, setup_logger
from mole.utils.data_utils import get_distance_matrix, get_clusters

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class RandomSimpleSplitter:
    """
    Simple random split performed by shuffling dataset and splitting according to indicated sizes. Useful for large
    datasets where DeepChem's RandomSplitter takes significantly longer.
    """

    def __init__(self):
        pass

    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0, **kwargs):
        split_prep = dataset.copy()
        split_prep = split_prep.sample(len(split_prep), random_state=seed)

        n_test = int(len(split_prep) * frac_test)
        n_val = int(len(split_prep) * frac_valid)

        test = split_prep.iloc[:n_test]
        val = split_prep.iloc[n_test:n_val]
        train = split_prep.iloc[n_test + n_val:]

        assert len(set(train.index).intersection(set(test.index))) == 0
        assert len(set(train.index).intersection(set(val.index))) == 0
        assert len(set(test.index).intersection(set(val.index))) == 0

        return train, val, test


class HierarchicalSplitter:
    """
        Performs Hierarchical Clustering on the dataset and splits data into model_developement/test sets
        where half of the test molecules are as far as possible from the training molecules .

        One half of the test set is made of the most dissimilar molecules (have weight -1) and the other half is
        a random collection of the dataset (average similar molecules , have weight +1)

        The train/val split set is designed from a random split of the obtained model development set
    """

    def __init__(self):
        pass

    def split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sim_th=0.5, evaluate=False, **kwargs):
        """
        Note !!! The simialrity threshold needs manual tuning, keep changing the threshold
        until you get a test fraction that is satisfying to you. The higher the threshold
        the more test molecules you'll have.


        Parameters
        ----------
        sim_th : The similarity threshold used to design clusters. It is the max similarity
                between train and test set.

        evaluate_th : Whether to evaluate the similarity threshold that you specified.
                    If True, it will just display the test fraction not return any split. (Default : False)
        """

        logger.info(f"Creating clusters for {dataset.X.shape[0]} molecules")
        clusters, cluster_counts = get_clusters(get_distance_matrix(dataset), 1 - sim_th)

        # The most dissimilar molecules at the given threshold
        diss_cluster = list(np.where(clusters != np.argmax(cluster_counts))[0])
        n_diss = len(diss_cluster)
        n_clusters = len(clusters)
        frac_diss = n_diss / n_clusters

        if evaluate:
            logger.info(f"{frac_diss * 100}% of the molecules are at most {sim_th} similar")
            assert (frac_diss < 0.5)
            return None, None, None
        else:
            specified_splitter = dc.splits.SpecifiedSplitter(valid_indices=np.array([], dtype="int32"),
                                                             test_indices=diss_cluster)
            rem_ds, dissim_ds = specified_splitter.train_test_split(dataset)
            # correct the fractions based on the chosen threshold
            frac_train = 1 - frac_valid - frac_test
            # Split the remaining molecules to get valid fractions
            rem_train_frac = frac_train / (1 - frac_diss)
            rem_val_frac = frac_valid / (1 - frac_diss)
            rem_test_frac = (frac_test - frac_diss) / (1 - frac_diss)

            assert (rem_train_frac + rem_val_frac + rem_test_frac == 1)

            random_splitter = dc.splits.RandomSplitter()
            # Split randomly the remaining molecules
            train_ds, val_ds, sim_ds = random_splitter.train_valid_test_split(rem_ds, frac_train=rem_train_frac,
                                                                              frac_valid=rem_val_frac,
                                                                              frac_test=rem_test_frac)

            # Annotate the two halves of the test set and merge them
            dissim_ds = dc.data.NumpyDataset(X=dissim_ds.X, y=dissim_ds.y, w=-np.ones(dissim_ds.X.shape[0]),
                                             ids=dissim_ds.ids)
            sim_ds = dc.data.NumpyDataset(X=sim_ds.X, y=sim_ds.y, w=np.ones(sim_ds.X.shape[0]), ids=sim_ds.ids)
            test_ds = dc.data.NumpyDataset.merge([dissim_ds, sim_ds])

            def similarity_to_train(x, sim_th):
                if x == 1:
                    return f"average similarity (>{sim_th})"
                else:
                    return f"low similarity (<{sim_th})"

            task = train_ds.get_task_names()[0]
            train_ds = pd.DataFrame({"SMILES": train_ds.ids, task: train_ds.y.flatten()})
            val_ds = pd.DataFrame({"SMILES": val_ds.ids, task: val_ds.y.flatten()})
            test_ds = pd.DataFrame({"SMILES": test_ds.ids, task: test_ds.y.flatten(),
                                    "Similarity_to_train": [similarity_to_train(x, sim_th) for x in test_ds.w]})

            return train_ds, val_ds, test_ds


AVAILABLE_SPLITTERS_DICT = {
    "RandomSplitter": dc.splits.RandomSplitter,
    "random": dc.splits.RandomSplitter,
    "ScaffoldSplitter": dc.splits.ScaffoldSplitter,
    "scaffold": dc.splits.ScaffoldSplitter,
    "MaxMinSplitter": dc.splits.MaxMinSplitter,
    "maxmin": dc.splits.MaxMinSplitter,
    "FingerprintSplitter": dc.splits.FingerprintSplitter,
    "fingerprint": dc.splits.FingerprintSplitter,
    "RandomStratifiedSplitter": dc.splits.RandomStratifiedSplitter,
    "random_stratified": dc.splits.RandomStratifiedSplitter,
    "HierarchicalSplitter": HierarchicalSplitter,
    "hierarchical": HierarchicalSplitter,
    "RandomSimpleSplitter": RandomSimpleSplitter,
    "random_simple": RandomSimpleSplitter

}


class Splitter:
    def __init__(self, split=None):
        """
        Parameters
        ----------
        split: str or dc.splits.Splitter class
            Splitter type to use. One of
              "RandomSplitter": ScaffoldSplitter, MaxMinSplitter, FingerprintSplitter, RandomStratifiedSplitter,
              HierarchicalSplitter,

             "scaffold", "maxmin", "butina", "random", "random_stratified", "hierarchical"
             or a dc.splits.Splitter class. By default, uses dc.splits.ScaffoldSplitter()
        """

        # Create a splitter from name or class
        if split is None:
            self.splitter = dc.splits.ScaffoldSplitter

        elif split in AVAILABLE_SPLITTERS_DICT:
            self.splitter = AVAILABLE_SPLITTERS_DICT.get(split)

        elif issubclass(split, dc.splits.Splitter):
            self.splitter = split

        else:
            error_msg = f"Type {type(split)} is not supported."
            logger.exception(error_msg)
            raise NotImplementedError(error_msg)

        self.splitter_name = self.splitter.__name__

        # Instantiate the splutter
        self.splitter = self.splitter()

        # This will store created data splits for further analysis
        self.splits = None

    def _convert_to_df(self, smiles, y):
        """
        Converts numpy arrays or pd.Series to pd.DataFrame
        """
        if not len(smiles) == len(y):
            error_msg = f"SMILES and targets are not of the same length, {len(smiles), len(y)}."
            logger.exception(error_msg)
            raise ValueError(error_msg)

        if type(smiles) is not pd.Series:
            smiles_name = "SMILES"
            index = np.arange(len(smiles))
        else:
            smiles_name = smiles.name
            index = smiles.index

        if type(y) is not pd.Series:
            y_name = "y"
        else:
            y_name = y.name

        df = pd.DataFrame({"ID": index, smiles_name: smiles, y_name: y})
        df = df.set_index("ID")

        self.smiles_name = smiles_name
        self.y_name = y_name

        return df

    def train_val_test_split(self, smiles: Union[pd.Series, np.ndarray],
                             y: Union[pd.Series, np.ndarray],
                             frac_test: float = 0.1,
                             frac_val: float = 0.1,
                             seed: int = 0,
                             shuffle: bool = True,
                             **kwargs):
        """
        Splits data into train test validation.

        Parameters
        ----------
        smiles: pd.Series, np.ndarray
        y: pd.Series, np.ndarray
        frac_test: float
        frac_val: float
        seed: int
        shuffle: bool
        kwargs: dict
            Additional arguments for individual splitters

        Returns
        -------
        train: pd.DataFrame
        val: pd.DataFrame
        test: pd.DataFrame

        """
        dataset = dc.data.DiskDataset.from_numpy(X=smiles.tolist(), y=y, w=np.zeros(len(smiles)), ids=smiles)
        train_idx, val_idx, test_idx = self.splitter.split(dataset,
                                                           frac_train=1 - frac_test - frac_val,
                                                           frac_valid=frac_val,
                                                           frac_test=frac_test,
                                                           seed=seed,
                                                           **kwargs)
        df = self._convert_to_df(smiles, y)
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        test = df.iloc[test_idx]

        if shuffle:
            train = train.sample(len(train), random_state=seed)
            val = val.sample(len(val), random_state=seed)
            test = test.sample(len(test), random_state=seed)

        if not all([indx not in set(train.index) for indx in list(test.index)]) or \
                not all([indx not in set(train.index) for indx in list(val.index)]):
            error_msg = "Faulty data splitting!"
            logging.exception(error_msg)
            raise UserWarning(error_msg)

        self.splits = {"train": train, "val": val, "test": test}

        return train, val, test

    def plot_target_distribution(self):
        """
        Plots target distribution for each train, test, val split
        """

        fig, axes = plt.subplots(figsize=(6 * len(self.splits), 5), ncols=len(self.splits), nrows=1)

        for i, (split_name, split_data) in enumerate(self.splits.items()):
            sns.kdeplot(split_data[self.y_name], fill=True, ax=axes[i])
            axes[i].set_title(f"{split_name} Data \n {self.y_name} Distribution")

        plt.show()

    def save_splits(self, save_dir: str, add_time_stamp: bool = False, index: Optional[bool] = False):
        """
        Save splits to a folder <path>

        Parameters
        ----------
        save_dir: str
            Path to a folder where to save splits. Splits are saved as csv files according to splitters' names.
        add_time_stamp: bool
            (Default False) Indicates whether to add a time stamp to the name of the folder.
        index: bool
            (Default False) Indicates whether to save indices of the samples.

        Returns
        -------

        """
        if add_time_stamp:
            stamp = get_time_stamp()
            path = os.path.join(save_dir, f"split_{stamp}")

        makedirs(path)

        for i, (split_name, split_data) in enumerate(self.splits.items()):
            split_data.to_csv(os.path.join(path, f"{split_name}.csv"), index=index)


class MultiSplitter:
    """
    Splits data using multiple splitters at once. For each splitting method, calls the Splitter class.
    """

    def __init__(self, splitters: list = None, include_tautomers: bool = False):
        """

        Parameters
        ----------
        splitters: list
            List of splitter names of splitter classes.
        include_tautomers: bool
            Indicates whether to generate separate splits also using tautomerized versions of molecules.
        """

        if splitters is None:
            default_splitters = ["RandomSplitter", "ScaffoldSplitter", "MaxMinSplitter", "FingerprintSplitter"]
            self.splitters = [Splitter(AVAILABLE_SPLITTERS_DICT[key]) for key in default_splitters]

        else:
            logger.info(f"Selected splitters: {splitters}.")

            self.splitters = []
            for key in splitters:
                splitter_class = AVAILABLE_SPLITTERS_DICT.get(key, None)
                if splitter_class is None:
                    logger.warning(f"{key} not found in available splitters.")
                else:
                    self.splitters.append(Splitter(splitter_class))

            if not len(self.splitters):
                error_msg = f"Did not find any valid splitter."
                logger.exception(error_msg)
                raise NotImplementedError(error_msg)

        self.splitter_names = [x.splitter_name for x in self.splitters]
        self.splits = {}
        self.include_tautomers = include_tautomers

    def generate_splits(self,
                        smiles: Union[pd.Series, np.ndarray],
                        y: Union[pd.Series, np.ndarray],
                        seed_val: int = None,
                        **kwargs):
        """

        Parameters
        ----------
        smiles: pd.Series, np.ndarray
        y: pd.Series, np.ndarray
        seed_val: int
            Allows to keep the test set the same and only vary random seed for train-validation split.
            If test set does not have to stay the same, consider using "seed" instead.
        kwargs:
            arguments passed to splitter.train_val_test_split method (seed, shuffle, frac_test, frac_val)

        Returns
        -------
        dict
            Dictionary with splitter names as keys and values being dictionaries with training splits:
             {"train": pd.DataFrame, "val":  pd.DataFrame,  "test":  pd.DataFrame}

        """

        for splitter_name, splitter in zip(self.splitter_names, self.splitters):
            logger.info(f"Splitting using {splitter_name}")

            if seed_val is not None:
                logger.info(f"Splitting {splitter_name} validation set with random seed: {seed_val}")
                logger.info(f"Args: {kwargs}")

                trainval, _, test = splitter.train_val_test_split(
                    smiles=smiles,
                    y=y,
                    seed=kwargs.get("seed", None),
                    shuffle=kwargs.get("shuffle", None),
                    frac_test=kwargs.get("frac_test", None),
                    frac_val=0
                )

                train, val, _ = splitter.train_val_test_split(
                    smiles=trainval.iloc[:, 0],
                    y=trainval.iloc[:, 1],
                    seed=kwargs.get("seed", None),
                    shuffle=kwargs.get("shuffle", None),
                    frac_test=0,
                    frac_val=kwargs.get("frac_val", None),
                )

                splitter.splits = {"train": train, "val": val, "test": test}
                logger.info(f"Train: {len(train)}   |   Val: {len(val)}    |    Test: {len(test)} \n\n")

            else:
                train, val, test = splitter.train_val_test_split(smiles, y, **kwargs)

            self.splits[splitter_name] = {"train": train,
                                          "val": val,
                                          "test": test}

            if self.include_tautomers:
                taut = TautomerGenerator()

                train_taut = taut(train.iloc[:, 0], train.iloc[:, 1], limit=20, standardize=True)
                val_taut = taut(val.iloc[:, 0], val.iloc[:, 1], limit=20, standardize=True)
                test_taut = taut(val.iloc[:, 0], val.iloc[:, 1], limit=20, standardize=True)

                self.splits[splitter_name].update({"train_taut": train_taut,
                                                   "val_taut": val_taut,
                                                   "test_taut": test_taut})
        return self.splits

    def plot_target_distribution(self, save_dir):
        """
        Plots target distribution for each train, test, val split for each splitting method.

        """

        fig, axes = plt.subplots(figsize=(18, 5 * len(self.splits)), ncols=3, nrows=len(self.splits), sharex="all")

        for i, (splitter_name, splitter) in enumerate(zip(self.splitter_names, self.splitters)):

            splits = self.splits[splitter_name]

            for j, (split_name, split_data) in enumerate(splits.items()):
                sns.kdeplot(split_data[splitter.y_name], fill=True, ax=axes[i, j])
                axes[i, j].set_title(f"{splitter_name} \n {split_name.title()} Data \n {splitter.y_name} Distribution")

        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "target_distribution.png"))

        else:
            plt.show()

    def save_splits(self, save_dir: str, add_time_stamp: bool = False, index: bool = None):
        """
        Save splits to a folder <path>

        Parameters
        ----------
        save_dir: str
            Path to a folder where to save splits. Splits are saved as csv files according to splitters' names.
        add_time_stamp: bool
            (Default False) Indicates whether to add a time stamp to the name of the folder.
        index: bool
            (Default False) Indicates whether to save indices of the samples.

        Returns
        -------

        """

        if add_time_stamp:
            stamp = get_time_stamp()
            save_dir = os.path.join(save_dir, f"split_{stamp}")

        for i, (splitter_name, splitter) in enumerate(zip(self.splitter_names, self.splitters)):
            path = os.path.join(save_dir, splitter_name)
            splitter.save_splits(path, index=index)

    def load_splits(self, save_dir: str, split_name: Optional[str] = None, index_col: Optional[str] = None):
        """
        Loads saves splits

        Parameters
        ----------
        save_dir: str
            Path to a folder where splits were saved (same path that was used in the save_splits method)
        split_name: str
            If only one split should be loaded, provide the split name here.
        index_col: str
            (Default None) Indicates whether to load one column as indices

        Returns
        -------
        dict
            Dictionary with splitter names as keys and values being dictionaries with training splits:
             {"train": pd.DataFrame, "val":  pd.DataFrame,  "test":  pd.DataFrame}

        """

        if split_name is not None:
            splitter_names = [split_name]
        else:
            splitter_names = self.splitter_names

        self.splits = {}

        for i, splitter_name in enumerate(splitter_names):
            path_to_split = os.path.join(save_dir, splitter_name)

            assert all([x in os.listdir(path_to_split) for x in
                        ["train.csv", "test.csv", "val.csv"]])

            train = pd.read_csv(os.path.join(path_to_split, "train.csv"), index_col=index_col)
            test = pd.read_csv(os.path.join(path_to_split, "test.csv"), index_col=index_col)
            val = pd.read_csv(os.path.join(path_to_split, "val.csv"), index_col=index_col)

            self.splits[splitter_name] = {"train": train,
                                          "val": val,
                                          "test": test}

        return self.splits
