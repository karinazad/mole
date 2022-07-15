import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix
from tqdm import tqdm


class ConformerGenerator:
    """
    Converts SMILES representations into 3D conformers using
    RDKit's distance geometry algorithm: http://rdkit.org/docs/Cookbook.html?highlight=conformers
    """

    def __init__(self,
                 smiles,
                 num_conformers=10,
                 minimize_energies=False,
                 **kwargs):
        """
        Generates and stores conformers for each SMILES string.
        Parameters
        ----------
        smiles: list, array, pd.Series
            Array of SMILES string representation.
        num_conformers: int
            Number of conformers to generate for each molecule
        kwargs
            Additional arguments to be passed to the AllChem.EmbedMultipleConfs function
        """
        if type(smiles) == str:
            smiles = [smiles]

        self.smiles = smiles
        self.num_conformers = num_conformers

        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.hmols = [Chem.AddHs(m) for m in self.mols]

        self._generate_conformers(num_conformers, minimize_energies, **kwargs)

    def _generate_conformers(self,
                             num_conformers,
                             minimize_energies,
                             **kwargs):
        """
        Generates conformers for each molecule and optionally minimizes
        their energies using MMFF force field.
        """

        # Generate conformers for molecules with added hydrogens
        for hmol in tqdm(self.hmols):
            AllChem.EmbedMultipleConfs(hmol, numConfs=num_conformers, **kwargs)

            if minimize_energies:
                AllChem.MMFFOptimizeMoleculeConfs(hmol)

        self.mols = [Chem.RemoveHs(hmol) for hmol in self.hmols]

    def get_conformers(self, include_hydrogens=False):
        """
        Returns generated conformers for each molecule.
        Parameters
        ----------
        include_hydrogens: bool
            Indicates whether to include hydrogen atoms.
        Returns
        -------
        conformers: nd.array
            An array of conformers of shape (num_molecules, num_conformers).
        """

        if include_hydrogens:
            conformers = [hmol.GetConformers() for hmol in self.hmols]

        else:
            conformers = [mol.GetConformers() for mol in self.mols]

        return conformers

    def get_3D_coords(self, include_hydrogens=False):
        """
        Returns 3D coordinates for each atom.
        The results are returned as a list of numpy arrays since different molecules
        have different number of atoms.
        Parameters
        ----------
        include_hydrogens: bool
            Indicates whether to include coordinates of hydrogen atoms.
        Returns
        -------
        conformers_3d_coords: list[nd.array]
            List of numpy arrays.
        """

        conformers = self.get_conformers(include_hydrogens)
        conformers_3d_coords = []

        for conformer_set in conformers:
            num_atoms = conformer_set[0].GetNumAtoms()
            conformer_positions = np.empty((self.num_conformers, num_atoms, 3))

            for i, c in enumerate(conformer_set):
                conformer_positions[i] = c.GetPositions()

            conformers_3d_coords.append(conformer_positions)

        return conformers_3d_coords

    def get_pairwise_rmsd(self):
        """
        Computes pairwise RMS values for each two conformers of a molecule (heavy atoms only).
        Returns
        -------
        conformers_rmsd: nd.array
            List of RMSD values from a flattened matrix.
        """
        conformers_rmsd = [AllChem.GetConformerRMSMatrix(mol) for mol in self.mols]
        return conformers_rmsd

    def get_pairwise_tfd(self):
        """
        Computes pairwise TFD values for each two conformers of a molecule (heavy atoms only).
        Returns
        -------
        conformers_rmsd: nd.array
            List of average TFD values.
        """

        conformers_tfd = []
        for mol in self.mols:
            try:
                tfd_mat = GetTFDMatrix(mol)
                conformers_tfd.append(tfd_mat)

            # Quick fix for an error that is sometimes called
            except IndexError:
                pass

        return conformers_tfd

    def get_rmds_ref_mol(self,
                         ref_mol,
                         mol_index):
        """
        Parameters
        ----------
        ref_mol
        mol_index
        Returns
        -------
        """
        num_failed_aligns = 0
        rmsd_list = []
        for i in range(self.num_conformers):

            try:
                qb = Chem.MolToMolBlock(self.mols[mol_index], confId=i)
                query_mol = Chem.MolFromMolBlock(qb)

                rmsd = Chem.rdMolAlign.AlignMol(ref_mol, query_mol)
                rmsd_list.append(rmsd)

            except RuntimeError as e:
                num_failed_aligns += 1
                continue

            # If "ValueError: Bad Conformer Id", there are no more conformers
            except ValueError:
                break

        print(f"Number of failed alignements: {num_failed_aligns} "
              f"(No sub-structure match found between the probe and query mol")
        print(f"Number of successful alignements: {len(rmsd_list)} \n\n")
        return rmsd_list

    def get_energies(self, mol_index=None, average=False):
        """
        """
        if mol_index is None:
            mols_to_compute_energies_for = self.hmols
        else:
            assert mol_index < len(self.hmols)
            mols_to_compute_energies_for = self.hmols[mol_index]

        energies = []

        for mol in tqdm(mols_to_compute_energies_for):
            AllChem.MMFFSanitizeMolecule(mol)
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)

            mol_energies = [
                AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id).CalcEnergy()
                for conf_id in range(len(mol.GetConformers()))
            ]

            if average:
                energies.append(np.mean(mol_energies))
            else:
                energies.append(mol_energies)

        return energies

    # def draw_conformers(self, index, num_conformers_to_show=5, mol_name=None):
    #     """
    #     Plots aligned conformers.
    #     Parameters
    #     ----------
    #     index: int
    #         Index of a molecule which conformers should be plotted.
    #     num_conformers_to_show: int
    #         Indicates number of conformers to be plotted.
    #     mol_name
    #     -------
    #     """
    #     assert index < len(self.mols), "Please provide a valid index"
    #
    #     if num_conformers_to_show > self.num_conformers:
    #         num_conformers_to_show = self.num_conformers
    #         print(f"Maximum number of conformers is {self.num_conformers}.")
    #
    #     mol = self.mols[index]
    #     AllChem.AlignMolConformers(mol)
    #
    #     p = py3Dmol.view(width=800, height=300)
    #
    #     for confId in range(num_conformers_to_show):
    #         mb = Chem.MolToMolBlock(mol, confId=confId)
    #         p.addModel(mb, 'sdf')
    #
    #     p.setStyle({'stick': {}})
    #     p.zoomTo()
    #     if mol_name is not None:
    #         print(mol_name)
    #     print(self.smiles[index])
    #     p.show()


# def draw_molecules(mols):
#     p = py3Dmol.view(width=800, height=300)
#     for mol in mols:
#         mb = Chem.MolToMolBlock(mol)
#         p.addModel(mb, 'sdf')
#
#     p.setStyle({'stick': {},
#                 'color': 'spectrum'})
#     p.zoomTo()
#     p.zoomTo()
#     p.show()


# The code below is adapted from:
# https://birdlet.github.io/2019/10/02/py3dmol_example/

# def MolTo3DView(mol, confid, style="stick", surface=False, opacity=0.5):
#     """Draw molecule in 3D
#     Args:
#     ----
#         mol: rdMol, molecule to show
#         size: tuple(int, int), canvas size
#         style: str, type of drawing molecule
#                style can be 'line', 'stick', 'sphere', 'carton'
#         surface, bool, display SAS
#         opacity, float, opacity of surface, range 0.0-1.0
#     Return:
#     ----
#         viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
#     """
#     assert style in ('line', 'stick', 'sphere', 'carton')
#     mblock = Chem.MolToMolBlock(mol, confId=confid)
#     viewer = py3Dmol.view(width=400, height=400)
#     viewer.addModel(mblock, 'mol')
#     viewer.setStyle({style: {}})
#     if surface:
#         viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
#     viewer.zoomTo()
#     return viewer