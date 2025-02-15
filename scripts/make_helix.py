#!/usr/bin/env python

import mdtraj as md
from PeptideBuilder import Geometry
import PeptideBuilder
import Bio.PDB
import argparse

def get_sequence(pdb_file):
    """
    Get the primary sequence of a protein from a topology or pdb file.
    To avoid issues with atom naming convention, we only consider residues
    with alpha carbons. This behavior might not be desirable in all cases.
    The default case is assuming that a PDB is made to compute S-alpha
    order parameter, in which case, only CA atoms need to be accounted for.
    """
    pdb = md.load(pdb_file)
    return "".join([pdb.top.atom(i).residue.code for i in pdb.top.select("name CA")])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description=("Make a PDB where all residues are in a helical conformation"
                    "from the input PDB"))
    
    parser.add_argument("--pdb", required=True, type=str,
                       help=("Reference PDB (Protein that will be transformed"
                             "into a perfect Helix)"))
    
    parser.add_argument("--phi", type=float, default = -60,
                       help="Phi angle (deg) to use for every residue")
    
    parser.add_argument("--psi", type=float, default = -40,
                       help="Psi angle (deg) to use for every residue")
    
    parser.add_argument("--out_pdb", required=False, type=str,
                       default=None, help="File name to give output (helix) PDB file")
    
    args = parser.parse_args()
    
    structure = None
    
    for i in iter(get_sequence(args.pdb)):

        residue = Geometry.geometry(i)
        residue.phi = args.phi
        residue.psi_im1 = args.psi

        if structure is None:
            structure=PeptideBuilder.initialize_res(residue)
        else:
            PeptideBuilder.add_residue(structure, residue)
    
    out_structure = Bio.PDB.PDBIO()
    out_structure.set_structure(structure)
    
    #check output file naming
    if args.out_pdb is None:
        file_name = f"{args.pdb[:-4]}_helix.pdb"
    
    else:
        if args.out_pdb[-4:]!=".pdb":
            file_name = f"{args.out_pdb}.pdb"
        else:
            file_name = args.out_pdb
    
    out_structure.save(file_name)
    

