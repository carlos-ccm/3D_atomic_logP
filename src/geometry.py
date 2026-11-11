#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: geometry.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Geometric and chemical helper functions (aromaticity detection,
# angles between vectors) used by the atomic descriptors.
# ----------------------------------------------------------

import warnings
import numpy as np
from rdkit import Chem


def detect_aromaticity(mol):
    '''
    Redefinition of aromaticity based on connectivity for not using GetIsAromatic() method of RDKit.
    '''
    aromatics = []

    #If there are rings
    ring_info = mol.GetRingInfo()
    vRings = ring_info.AtomRings()
    #print(mol.GetProp("_Name"),vRings)
    if vRings:
        for ring in vRings:
            #print("ring",ring)
            bAr = False
            value = get_ring_atom_sum(mol,ring)

            #Detect rings with sp3 carbon
            sp3 = False
            for at_id in ring:
                at = mol.GetAtomWithIdx(at_id)
                if at.GetSymbol() == "C":
                    if at.GetHybridization() == Chem.HybridizationType.SP3:
                        sp3 = True
                        break
            if sp3:
                continue

            # Planar rings formed "outside" double bonds
            #print(mol.GetProp("_Name"),value,len(ring))
            if value >= len(ring) and value <= 2 * len(ring):
                for at_id in ring:
                    at = mol.GetAtomWithIdx(at_id)
                    for bond in at.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE or bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                            neighbor_id = bond.GetOtherAtomIdx(at_id)
                            if neighbor_id not in ring:
                                #print("test")
                                bAr = True
                                break
                    if bAr:
                        break

            # Pure planar aromatic ring
            if bAr != True and value == 12 and len(ring) == 6:
                bAr = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)

            # For anionic triazole ring and other rings:
            if bAr != True and value >= (len(ring) + 3):
                bAr = True
            if bAr:
                for at_id in ring:
                    aromatics.append(at_id)
    return aromatics

def get_ring_atom_sum(mol, ring):
    value = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()

        # Carbon
        if atomic_num == 6:
            if degree == 3:
                value += 2
            elif degree == 4:
                value -= 2

        # Nitrogen
        elif atomic_num == 7:
            if degree <= 3:
                value += 2

        # Oxygen
        elif atomic_num == 8:
            if degree == 2:
                value += 1

        # Phosphorus
        elif atomic_num == 15:
            if degree == 2:
                value += 2
            elif degree == 3:
                value += 1
            elif degree >= 4:
                value -= 1

        # Sulfur
        elif atomic_num == 16:
            if degree == 2:
                value += 1
            elif degree >= 3:
                value -= 1

    return value

def angle_vectors (u,v):
    escalar_prod = np.dot(u,v)
    module_u =np.linalg.norm(u)
    module_v =np.linalg.norm(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        angle = np.degrees(np.arccos(escalar_prod/(module_u * module_v)))
    return angle
