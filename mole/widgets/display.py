import pandas as pd
import numpy as np
import rdkit
from IPython.display import display, HTML
from ipywidgets import widgets
from ipywidgets import interact, Layout, Button


def df_to_html(df):
    pd.set_option("display.max_rows", None)

    df_display = df.copy()

    # Reindex if not
    if len(np.unique(list(df_display.index))) != len(df):
        index_name = df_display.index.name if df_display.index.name is not None else "ID"
        df_display[index_name] = df_display.index
        df_display.index = np.arange(len(df_display))

    return HTML(
        "<div style='height: 500px; overflow: auto; width: fit-content'>" + df_display.style.render() + "</div>")


def browse_images(mols):
    n = len(mols)

    def view_image(Index):
        img = rdkit.Chem.Draw.MolsToGridImage([mols[Index]], molsPerRow=5, subImgSize=(400, 400), returnPNG=True)

        smiles = rdkit.Chem.MolToSmiles(mols[Index])
        indicator_text = widgets.Label(value=f"Index = {Index}, \n SMILES = {smiles}",
                                       layout=widgets.Layout(height="4", width="auto"), style={"font-weight": "bold"})

        display(img)
        display(indicator_text)

    interact(view_image,
             Index=widgets.IntSlider(min=0, max=n - 1, step=1, value=0, layout=Layout(width='50%', height='80px'),
                                     style={"handle_color": "green"}))


def browse_annotations_grouped(df):
    df = df.copy()
    n = len(np.unique(df["Molecule ID"]))

    def view_single_annotation(molecule_id):
        rows = df[df["Molecule ID"] == molecule_id]

        smis = rows["SMILES"]
        mols = [rdkit.Chem.MolFromSmiles(s) for s in smis]

        matched_atoms = rows["Matched Atoms"]
        matched_atoms = [m if m != "no match" else [] for m in matched_atoms]

        img = rdkit.Chem.Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(400, 400),
                                              highlightAtomLists=matched_atoms, returnPNG=True)

        display(img)
        display(df_to_html(rows))

    slider = widgets.IntSlider(min=0, max=n - 1, step=1, value=0, layout=Layout(width='100%', height='80px'),
                               style={"handle_color": "green"}, description="Molecule ID")

    interact(view_single_annotation, molecule_id=slider)

