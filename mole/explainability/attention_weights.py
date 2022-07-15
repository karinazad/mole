"""
This module is used for visualizing AttentiveFP's attention weights in the readout layer.

For more information about the readout phase and the attention weights see:
Xiong et al. (2019) Pushing the Boundaries of Molecular Representation for Drug Discovery
with the Graph Attention Mechanism (https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)

"""
from typing import Union

import numpy as np
import torch
import matplotlib
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display


def draw_atom_weights(smiles: str, atom_weights: Union[torch.Tensor, np.ndarray, list]):
    """
    Plots a molecule where radius of circles around atoms corresponds to provided atom weights.

    Parameters
    ----------
    smiles: str
        SMILES of a molecule to be plotted
    atom_weights: torch.Tensor, np.ndarrray or list
        an array of floats with the same length as SMILES

    """
    assert len(smiles) == len(atom_weights), f"Lengths of SMILES and atom weights " \
                                             f"do not match: {len(smiles), len(atom_weights)} "
    number_of_nodes = len(atom_weights)

    # Normalize weights
    min_value = torch.min(atom_weights)
    max_value = torch.max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)

    # Convert the weights to atom colors and radii
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = matplotlib.cm.get_cmap('Oranges')
    plt_colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i].data.item()) for i in range(number_of_nodes)}
    atom_radii = {i: atom_weights[i].data.item() for i in range(number_of_nodes)}

    # Draw molecule
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    drawer.SetFontSize(1)
    op = drawer.drawOptions()

    # Add highlights according to atom weights
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(number_of_nodes), highlightBonds=[],
                        highlightAtomColors=atom_colors, highlightAtomRadii=atom_radii)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    display(SVG(svg))


def get_attention_weights(g: dc.feat.graph_data.GraphData,
                          model: dc.models.AttentiveFPModel):
    """
    Get attention weights from a DeepChem's AttentiveFP model by accessing its internal DGL model.

    Parameters
    ----------
    g: dc.feat.graph_data.GraphData
        DeepChem's graph object
    model: dc.models.AttentiveFPModel
        Trained AttentiveFP model

    Returns
    ----------
    node_weights: torch.Tensor
        Attention weights (from the readout phase of AttentiveFP) for each node in the graph

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Access the DGL model (which deepchem builds upon)
    dgl_model = model.model.model

    # Since we are using DGL model, convert the deepchem's graph to a DGL graph
    g_dgl = g.to_dgl_graph().to(device)

    # DGL model take in node features and edge features separately
    node_feats = torch.Tensor(g.node_features).to(device)
    edge_feats = torch.Tensor(g.edge_features).to(device)

    # Set get_node_weight=True to get attention node weights in a forward pass
    _, node_weights = dgl_model(g_dgl, node_feats, edge_feats, get_node_weight=True)

    node_weights = node_weights[1].detach()

    return node_weights
