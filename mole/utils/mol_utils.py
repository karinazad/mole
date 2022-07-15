import rdkit
from rdkit import Chem

def get_molblock_from_smiles(s):
    try:
        mol = rdkit.Chem.MolFromSmiles(s)
        molblock = rdkit.Chem.MolToMolBlock(mol)
    except:
        molblock = ""
    return molblock


def check_smiles_issues(smiles):
    molblocks = smiles.apply(get_molblock_from_smiles)
    issues = molblocks.apply(check_molblock)
    return issues


def standardize_molecules(smiles):
    try:
        import chembl_structure_pipeline
    except:
        raise UserWarning("This module requires chembl_structure_pipeline package")

    molblocks = smiles.apply(get_molblock_from_smiles)
    standardized = molblocks[molblocks != ""].apply(lambda x:
                                                    chembl_structure_pipeline.standardizer.standardize_molblock(x))
    standardized = standardized.apply(Chem.MolFromMolBlock)
    standardized = standardized.apply(Chem.MolToSmiles)
    return standardized


def parentize_molecules(smiles):
    try:
        import chembl_structure_pipeline
    except:
        raise UserWarning("This module requires chembl_structure_pipeline package")

    molblocks = smiles.apply(get_molblock_from_smiles)
    parent = molblocks[molblocks != ""].apply(lambda x:
                                              chembl_structure_pipeline.standardizer.get_parent_molblock(x)[0])
    parent = parent.apply(Chem.MolFromMolBlock)
    parent = parent.apply(Chem.MolToSmiles)
    return parent


def check_molblock(molblock):
    try:
        import chembl_structure_pipeline
    except:
        raise UserWarning("This module requires chembl_structure_pipeline package")

    if molblock == "":
        return (8, 'Invalid SMILES string: Failed to convert to a molecule.'),
    else:
        return chembl_structure_pipeline.checker.check_molblock(molblock)


def smiles_to_smarts(smiles):
    molblocks = smiles.apply(get_molblock_from_smiles)
    mols = molblocks.apply(Chem.MolFromMolBlock)
    smarts = mols.apply(Chem.rdmolfiles.MolToSmarts)

    return smarts
