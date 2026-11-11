#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: descriptors.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# 3D atomic descriptors used to featurize each atom: electrostatic
# (Gasteiger) and van der Waals environment, and bonded neighbours.
# ----------------------------------------------------------

import numpy as np
from src.geometry import angle_vectors

def environment_interactions(mol,self_at,self_id,self_coords,neighbor_indices_2,neighbor_indices_3):
    element_radius = {
            6: 1.70,    # Carbon
            1: 1.20,    # Hydrogen
            7: 1.55,    # Nitrogen
            8: 1.52,    # Oxygen
            16: 1.80,   # Sulfur
            9: 1.47,    # Fluorine
            17: 1.75,   # Chlorine
            35: 1.85,   # Bromine
            15: 1.80,   # Phosphorus
            53: 1.98,   # Iodine
            5: 1.91,    # Boron (From Alvarez'13)
            4: 1.98,    # Beryllium (From Alvarez'13)
            13: 2.25,   # Aluminium (From Alvarez'13)
            30: 1.39,   # Zinc
            2: 1.40,    # Helium
            29: 1.40,   # Copper
            10: 1.54,   # Neon
            80: 1.55,   # Mercury
            48: 1.58,   # Cadmium
            28: 1.63,   # Nickel
            46: 1.63,   # Palladium
            79: 1.66,   # Gold
            47: 1.72,   # Silver
            12: 1.73,   # Magnesium
            78: 1.75,   # Platinum
            3: 1.82,    # Lithium
            33: 1.85,   # Arsenic
            92: 1.86,   # Uranium
            31: 1.87,   # Gallium
            18: 1.88,   # Argon
            34: 1.90,   # Selenium
            49: 1.93,   # Indium
            81: 1.96,   # Thallium
            36: 2.02,   # Krypton
            82: 2.02,   # Lead
            52: 2.06,   # Tellurium
            14: 2.10,   # Silicon
            74: 2.10,   # Tungsten
            54: 2.16,   # Xenon
            50: 2.17,   # Tin
            11: 2.27,   # Sodium
            19: 2.75    # Potassium
        }
    gasteiger_mod = 0
    vdw_interaction = 0
    for at in mol.GetAtoms():
        try:
            at_id = at.GetIdx()
            if at_id == self_id or at_id in neighbor_indices_2:
                continue
            else:
                element = mol.GetAtomWithIdx(at_id).GetAtomicNum()

                coords_i = mol.GetConformer().GetAtomPosition(at_id)
                dist = (coords_i.x-self_coords.x)**2+(coords_i.y-self_coords.y)**2+(coords_i.z-self_coords.z)**2 #SQRT is heavy remove for dist < 16
                if dist < 12.25:
                    gasteiger_contribution = round(at.GetDoubleProp('_GasteigerCharge'),6)
                    vdw_contribution = 4/3 * np.pi * element_radius.get(element) ** 3
                    if len(self_at.GetNeighbors()) != 1:
                        if at_id in neighbor_indices_3:
                            continue
                        else:
                            gasteiger_mod += gasteiger_contribution
                            vdw_interaction += vdw_contribution

                    else:
                        #Angular restriction
                        at_neigh = self_at.GetNeighbors()[0]

                        coords_neigh = mol.GetConformer().GetAtomPosition(at_neigh.GetIdx())
                        vector_i = np.array(self_coords - coords_neigh)
                        vector_potential_electrostatic_contributor = np.array(coords_i - coords_neigh)
                        alpha = angle_vectors(vector_i,vector_potential_electrostatic_contributor)
                        if alpha < 60:
                            gasteiger_mod += gasteiger_contribution
                            vdw_interaction += vdw_contribution
        except:
            continue

    return(gasteiger_mod,vdw_interaction)

def bonded_neighbors(at,mol):
        self_id = at.GetIdx()
        neighbor_list = []
        neighbor_indices = [neighbor.GetIdx() for neighbor in at.GetNeighbors()]
        for n in (neighbor_indices):
            neighbor_list.append(mol.GetAtomWithIdx(n))

        for at_2 in neighbor_list:
            new_neighbor_indices = [neighbor.GetIdx() for neighbor in at_2.GetNeighbors()]
            for i in new_neighbor_indices:
                if i != self_id:
                    neighbor_indices.append(i)
        neighbor_indices_2 = list(set(neighbor_indices))

        for n_2 in (neighbor_indices):
            neighbor_list.append(mol.GetAtomWithIdx(n_2))

        for at_3 in neighbor_list:
            new_neighbor_indices = [neighbor.GetIdx() for neighbor in at_3.GetNeighbors()]
            for i in new_neighbor_indices:
                if i != self_id:
                    neighbor_indices.append(i)
        neighbor_indices_3 = list(set(neighbor_indices))
        return neighbor_indices_2,neighbor_indices_3