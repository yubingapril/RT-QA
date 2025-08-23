import os
import numpy as np
from Bio import PDB
from joblib import Parallel, delayed
from itertools import combinations
import pandas as pd
from scipy.spatial import cKDTree


class cal_interface_atom():
    pdb_parser = PDB.PDBParser(QUIET=True)

    def __init__(self, inputfile, cut=8):
        self.inputfile = inputfile
        self.cut = cut

    # Extract all atom information, not just CA atoms
    def extract_atoms_info(self):
        structure = self.pdb_parser.get_structure('structure', self.inputfile)
        
        atoms_info = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Exclude heteroatoms like water
                        for atom in residue:
                            atom_name = atom.get_name()
                            chain_id = chain.id
                            res_id = residue.get_id()[1]
                            ins_code = residue.get_id()[2].strip()  # Get insertion code
                            coords = atom.get_coord()
                            res_name = residue.get_resname()
                            atoms_info.append((chain_id, res_id, res_name, ins_code, atom_name, coords))

        return atoms_info

    # Calculate interface atoms (not just CA)
    def calculate_interface_index(self, atoms_info):
        interface_index = set()
        
        # Extract coordinates and chain IDs
        coords = np.array([atom[5] for atom in atoms_info])  # The 5th element is the coordinate
        chain_ids = np.array([atom[0] for atom in atoms_info])  # The 1st element is the chain ID
        kdtree=cKDTree(coords)
        pairs=kdtree.query_pairs(self.cut)

        # Iterate over all atom pairs
        for i,j in pairs:
            if chain_ids[i] == chain_ids[j]:  # Skip atom pairs that belong to the same chain
                continue
            interface_index.add(atoms_info[i][:-1]+(tuple(atoms_info[i][-1])))
            interface_index.add(atoms_info[j][:-1]+(tuple(atoms_info[j][-1])))


        return sorted(interface_index, key=lambda x: (x[0], int(x[1]), x[3]))  # Sort by chain ID, residue ID, and insertion code

    # Write interface atom information to a file
    def write_interface_info(self, atoms_info, outfile):
        interface_info = self.calculate_interface_index(atoms_info)
        # print(len(interface_info))
        with open(outfile, 'w') as f:
            # print(interface_info)
            for chain_id, res_id, res_name, ins_code, atom_name, coord_1,coord_2,coord_3 in interface_info:
                # coord_str = ' '.join(map(str, coord))
                if atom_name[0]=='H' or atom_name[0]=='D':
                    continue
                if ins_code == '':  # No insertion code
                    f.write(f"c<{chain_id}>r<{res_id}>R<{res_name}>A<{atom_name}> {coord_1:.3f} {coord_2:.3f} {coord_3:.3f}\n")
                else:  # With insertion code
                    f.write(f"c<{chain_id}>r<{res_id}>i<{ins_code}>R<{res_name}>A<{atom_name}> {coord_1:.3f} {coord_2} {coord_3:.3f}\n")

    # High-level function to extract and write interface atom info
    def find_and_write(self, outfile):
        atoms_info = self.extract_atoms_info()
        # print(len(atoms_info))
        self.write_interface_info(atoms_info, outfile)


def interface_batch(pdb_dir,atom_coor_dir,n):
    model_list = [file.split('.')[0] for file in os.listdir(pdb_dir)]
    Parallel(n_jobs=n)(
            delayed(lambda model: cal_interface_atom(os.path.join(pdb_dir, f"{model}.pdb"), cut=6.8).find_and_write(os.path.join(atom_coor_dir, f"{model}.txt")))(model) for model in model_list
        )    

    

