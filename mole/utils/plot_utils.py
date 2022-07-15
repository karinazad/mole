import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem


def plot_distributions(values, save_dir=None, title=None, as_plt=False, **kwargs):

    fig, axes = plt.subplots(figsize=(6 * len(values), 5), ncols=len(values), nrows=1, sharex=True)

    if len(values) == 1:
        sns.kdeplot(np.array(values).flatten(), fill=True)

    else:
        if type(values) is dict:
            for i, (key, value) in enumerate(values.items()):
                if as_plt:
                    axes[i].hist(np.array(value).flatten(), fill=True, bins=100, **kwargs)

                sns.kdeplot(np.array(value).flatten(), fill=True, ax=axes[i], **kwargs)
                axes[i].set_title(key)

        else:
            for i, value in enumerate(values):
                if as_plt:
                    axes[i].hist(np.array(value).flatten(), fill=True, bins=100, **kwargs)
                sns.kdeplot(np.array(value).flatten(), fill=True, ax=axes[i],  **kwargs)
                axes[i].set_title(i)

    plt.tight_layout()

    if title is not None:
        fig.suptitle(title)

    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

    return fig


def get_tsne_embedding(smiles):
    from sklearn.manifold import TSNE

    def fp2arr(fp):
        arr = np.zeros((0,))
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols]
    X = np.asarray([fp2arr(fp) for fp in fps]).astype(np.float32)
    embeddings = TSNE(init='pca', random_state=794, verbose=0).fit_transform(X)

    return embeddings
