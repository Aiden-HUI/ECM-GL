# proprocessor.py
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import Optional, Tuple

class MolecularPreprocessor:
    def __init__(self):

        self.atom_features = {
            'atomic_num': list(range(1, 119)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, 0, 1],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
            'is_aromatic': [0, 1]
        }

        self.bond_features = {
            'bond_type': [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'stereo': [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE
            ],
            'is_conjugated': [0, 1]
        }

    def _get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        features = []

        features += [int(atom.GetAtomicNum() == x) for x in self.atom_features['atomic_num']]

        features += [int(atom.GetDegree() == x) for x in self.atom_features['degree']]

        features += [int(atom.GetFormalCharge() == x) for x in self.atom_features['formal_charge']]

        features += [int(atom.GetHybridization() == x) for x in self.atom_features['hybridization']]

        features += [int(atom.GetIsAromatic())]

        return np.array(features, dtype=np.float32)

    def _get_bond_features(self, bond: Chem.Bond) -> np.ndarray:

        features = []

        features += [int(bond.GetBondType() == x) for x in self.bond_features['bond_type']]

        features += [int(bond.GetStereo() == x) for x in self.bond_features['stereo']]

        features += [int(bond.GetIsConjugated())]

        return np.array(features, dtype=np.float32)

    def _generate_3d_conformation(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            return coords.astype(np.float32)
        except:
            return None

    def process_smiles(self, smiles: str) -> Optional[Tuple[Data, np.ndarray]]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            atom_features_list = [self._get_atom_features(atom) for atom in mol.GetAtoms()]

            atom_features = np.array(atom_features_list, dtype=np.float32)

            edge_indices = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_indices.extend([[i, j], [j, i]])

                feats = self._get_bond_features(bond)
                edge_features_list.extend([feats, feats.copy()])

            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            edge_features = np.array(edge_features_list, dtype=np.float32)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

            coords_3d = self._generate_3d_conformation(Chem.AddHs(mol))
            if coords_3d is None:
                return None

            original_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
            coords_3d = coords_3d[original_atom_indices]

            data_2d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            return (data_2d, coords_3d)

        except Exception as e:
            print(f"Error processing {smiles}: {str(e)}")
            return None
