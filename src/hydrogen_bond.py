#!/usr/bin/env python
# ----------------------------------------------------------
# CBDD - Torribera, Universitat de Barcelona
#
# File: hydrogen_bond.py
# Author: Carlos Cruz-Marín
# ----------------------------------------------------------
# Hydrogen-bond descriptors: identification of acceptors/donors and
# detection of intramolecular hydrogen bonds.
# ----------------------------------------------------------

import numpy as np
from rdkit import Chem, RDLogger
from src.geometry import detect_aromaticity, angle_vectors

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def getHB_acceptors_donors(m):
    '''
    Defines the hydrogen-bond acceptors and donors
    based on the degree of the atoms and its connectivity. Returns an array with the indexes of the acceptors in the molecule.
    '''
    HB_acceptors= []
    HB_donors= []
    aromatics = detect_aromaticity(mol=m)
    for at in m.GetAtoms():
        # Nitrogen atoms
        if at.GetSymbol() =="N":
            if at.GetDegree() == 1:
                HB_acceptors.append(at.GetIdx())
                for neigh in at.GetNeighbors():
                    if neigh.GetSymbol() == "H":
                        HB_donors.append(at.GetIdx())
                continue
            elif at.GetDegree() == 2:
                HB_acceptors.append(at.GetIdx())
                for neigh in at.GetNeighbors():
                    if neigh.GetSymbol() == "H":
                        HB_donors.append(at.GetIdx())
                        continue
                continue
            elif at.GetDegree() == 3:
                #print(m.GetProp("_Name"),at.GetIdx())
                db, NO2, da, CO, SO = 0, 0, 0, 0, 0
                for neigh in at.GetNeighbors():
                    if neigh.GetSymbol() == "H":
                        HB_donors.append(at.GetIdx())
                        continue
                for neighbor in at.GetNeighbors():
                    if m.GetBondBetweenAtoms(at.GetIdx(),neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE or m.GetBondBetweenAtoms(at.GetIdx(),neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        db += 1
                    if CO == 0 and neighbor.GetSymbol() == "C":
                        for neighbor_2 in neighbor.GetNeighbors():
                            if neighbor_2.GetSymbol() == "O" and neighbor_2.GetDegree() == 1:
                                CO += 1
                                continue
                    if SO == 0 and neighbor.GetSymbol() == "S":
                        for neighbor_2 in neighbor.GetNeighbors():
                            if neighbor_2.GetSymbol() == "O" and neighbor_2.GetDegree() == 1:
                                SO += 1
                                continue
                    if neighbor.GetSymbol() == "O":
                        NO2 += 1

                    ''' Checking for each of the bonds of a N atom with 3
                        bonds It is connected to a C atom that is any of:
                        int ar1; //Pure aromatic atom such as heavy atoms in benzene and pyridine
                            Tots anells en ressonancia
                        int ar2; //Other planar ring, except ar6
                            Si no te enllasos dobles alifatic, planar tot double bonds
                        int ar3; //Planar ring atom with one or more external double bonds
                            els bonds del ring amb atom amb double bond a fora son single
                        int ar4;//Other
                        int ar5;//Pure aliphatic ring atom, carbons with single bond
                        int ar6;//Anionic triazole ring anell '''
                    #print(aromatics)
                    if da == 0 and neighbor.GetSymbol() == "C" and neighbor.GetIdx() in aromatics:
                        da += 1

                '''Checking if the N with 3 bonds is NOT
                    int ar1;//Pure aromatic atom such as heavy atoms in benzene and pyridine
                    int ar2;//Other planar ring, except ar6
                    int ar3;//Planar ring atom with one or more external double bonds
                    int ar4;//Other
                    int ar5;//Pure aliphatic ring atom, carbons with single bond
                    int ar6;//Anionic triazole ring '''

                if CO == 0 and db == 0 and SO == 0 and da == 0 and at.GetIdx() not in aromatics:
                    HB_acceptors.append(at.GetIdx())
                for neighbor in at.GetNeighbors():
                    if NO2 == 2 and neighbor.GetSymbol() == "O":
                        HB_acceptors.append(neighbor.GetIdx())

            elif at.GetDegree() == 4:
                for neigh in at.GetNeighbors():
                    if neigh.GetSymbol() == "H":
                        HB_donors.append(at.GetIdx())
                        continue

        # Oxygen atoms
        elif at.GetSymbol() == "O":
            if at.GetDegree() == 1:
                for neighbor in at.GetNeighbors():
                    if neighbor.GetSymbol() == "C" or neighbor.GetSymbol() == "P" or neighbor.GetSymbol() == "S":
                        HB_acceptors.append(at.GetIdx())
            elif at.GetDegree() == 2:
                OH = 0
                donor = False
                for neighbor in at.GetNeighbors():
                    if neighbor.GetSymbol() == "H":
                        donor = True
                    if m.GetBondBetweenAtoms(at.GetIdx(),neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE or m.GetBondBetweenAtoms(at.GetIdx(),neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        OH += 1
                if donor == True and OH == 0:
                    HB_acceptors.append(at.GetIdx())
                    HB_donors.append(at.GetIdx())
                    continue

        # Fluorine atoms
        elif at.GetSymbol() == "F":
            HB_acceptors.append(at.GetIdx())
            continue
    HB_donors = list(set(HB_donors))
    return HB_acceptors, HB_donors

def HB_detector(mol,at_self,coords_H,potential_acceptors):
    '''
    This function returns if there is an internal Hydrogen Bond given the molecule, the atom, the coordinates of the Hydrogen
    and the acceptors present in the molecule.
    The definition of the Hydrogen Bond is based on the distance between the donor and the accepto (<3.5), the angle A-D-H <45
    degrees.
    Also a detector of potential H close to the polar H for not assigning a Hydrogen Bond.
    '''
    hb = 0
    if at_self.GetSymbol() == "H":
        for at in at_self.GetNeighbors():
            if at.GetSymbol() == 'C':
                continue
            else:
                at_id = at.GetIdx()
                coords_D = mol.GetConformer().GetAtomPosition(at_id)
                for at_potential_acceptor in potential_acceptors:
                    condition = False
                    coords_A = mol.GetConformer().GetAtomPosition(at_potential_acceptor) #coordinates of the potential acceptor
                    dist = ((coords_A.x-coords_D.x)**2+(coords_A.y-coords_D.y)**2+(coords_A.z-coords_D.z)**2)

                    if dist > 12.25:
                        continue
                    else:
                        at_potential_acceptor = mol.GetAtomWithIdx(at_potential_acceptor)
                        for neigh_acceptor in at_potential_acceptor.GetNeighbors():
                            if neigh_acceptor.GetSymbol() == "H": #Check how to redefine this function
                                coords_H_Acceptors = mol.GetConformer().GetAtomPosition(neigh_acceptor.GetIdx())
                                dist = ((coords_H.x-coords_H_Acceptors.x)**2+(coords_H.y-coords_H_Acceptors.y)**2+(coords_H.z-coords_H_Acceptors.z)**2)
                                if dist < 1.96:
                                    condition = True
                        if condition == False:
                            vector_H = np.array(coords_H - coords_D)
                            vector_A = np.array(coords_A - coords_D)
                            angle_alfa = angle_vectors(vector_A,vector_H)
                            if angle_alfa < 45:
                                hb = 1
    return hb