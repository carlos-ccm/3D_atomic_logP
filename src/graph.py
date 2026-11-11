#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: graph.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Conversion of 3D molecules into graphs (atom and bond features)
# for the GNN, and construction of the training dataset.
# ----------------------------------------------------------

import torch
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit.Chem import AllChem,rdchem
from rdkit.Chem.rdmolops import Get3DDistanceMatrix

from src.descriptors import bonded_neighbors, environment_interactions
from src.hydrogen_bond import HB_detector, getHB_acceptors_donors

def atom_features(atom, mol, at_coords, potential_acceptors):
    Z = atom.GetAtomicNum()
    g, v = environment_interactions(mol, atom, atom.GetIdx(), at_coords, *bonded_neighbors(atom, mol))
    hb = HB_detector(mol, atom, at_coords, potential_acceptors)

    # Standard RDKit atomic features
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybridization = int(atom.GetHybridization())
    total_hs = atom.GetTotalNumHs()
    is_aromatic = int(atom.GetIsAromatic())
    is_in_ring = int(atom.IsInRing())
    mass = atom.GetMass()

    return torch.tensor([
        g, v, hb,
        degree, formal_charge, hybridization,
        total_hs, is_aromatic, is_in_ring, mass
    ], dtype=torch.float),Z


def mol_to_graph(mol, with_targets=True):
    """
    Builds the PyTorch Geometric graph of a 3D molecule with explicit hydrogens.

    with_targets=True  (training): reads the atomic logP target.
        Returns None if any atom lacks a valid target.
    with_targets=False (prediction): no targets are read.
    """
    dist_matrix = Get3DDistanceMatrix(mol)
    AllChem.ComputeGasteigerCharges(mol)

    features, targets, atomic_numbers, edges = [], [], [], [[], []]
    potential_acceptors = getHB_acceptors_donors(mol)[0]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        coords = mol.GetConformer().GetAtomPosition(idx)
        feat, Z = atom_features(atom, mol, coords, potential_acceptors)
        features.append(feat)
        atomic_numbers.append(Z)
        if with_targets:
            try:
                values = atom.GetProp("molFileValue").split()
                targets.append([float(values[1])])  # atomic logP
            except (KeyError, IndexError, ValueError):
                return None

    edge_attrs = []
    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Add edge index in both directions
        edges[0].extend([i, j])
        edges[1].extend([j, i])
        dist = dist_matrix[i][j]

        # Encode bond features
        bond_type = bond.GetBondType()
        bond_feats = [
            int(bond_type == rdchem.BondType.SINGLE),
            int(bond_type == rdchem.BondType.DOUBLE),
            int(bond_type == rdchem.BondType.TRIPLE),
            int(bond_type == rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            dist]

        edge_attrs.extend([bond_feats, bond_feats])
    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)
    data = Data(
        x=torch.stack(features),
        edge_index=torch.tensor(edges, dtype=torch.long),
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        name=mol.GetProp("_Name") if mol.HasProp("_Name") else "mol",
        edge_attr=edge_attr_tensor)
    if with_targets:
        data.y = torch.tensor(targets, dtype=torch.float)
    return data


def build_dataset(sdf_file):
    """Builds the list of training graphs from an SDF file (molecules without valid targets are skipped)."""
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    dataset = []
    for mol in tqdm(suppl, desc="Loading molecules"):
        if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumBonds() == 0:
            continue
        g = mol_to_graph(mol, with_targets=True)
        if g is not None:
            dataset.append(g)
    return dataset
