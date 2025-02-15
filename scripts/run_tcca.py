import os
import numpy as np
from writhe_tools.utils import save_dict, makedirs, lsdir, reindex_list
from writhe_tools.writhe import Writhe
from writhe_tools.md_tools import ResidueDistances
import mdtraj as md
from writhe_tools.tcca import _tcca_scores
from writhe_tools.utils.filing import makedirs


root = "./"
indices = md.load("./PaaA2.pdb").top.select("name CA")
traj = md.load("./PaaA2-a99SBdisp-Traj.dcd", top="./PaaA2.pdb", atom_indices=indices)
distances = ResidueDistances(index_0=np.arange(traj.n_residues), traj=traj)
distances.save("./paaa2_ca_distances.pkl")
del traj

distances_path = makedirs(f"{root}/distances")
lags = np.array([1, 5] + list(range(11, 111, 10)))

_tcca_scores(distances.distances, lags, 20, path=distances_path)
del distances

writhe_path = makedirs(f"{root}/writhe")
writhe_files = lsdir("./", keyword=["writhe", "pkl"], indexed=True)

for length, writhe_file in zip([1, 2, 3, 4, 5], writhe_files):
    
    writhe = Writhe.load(writhe_file).writhe_features
    _tcca_scores(writhe, lags, 20, path=f"{writhe_path}/length_{length}")


writhe_pairs = [[0, 1], [0, 2], [0, 2, 4], [0, 1, 2]]

for pair in writhe_pairs:
    writhe = np.concatenate([Writhe.load(i).writhe_features for i in reindex_list(writhe_files, pair)], axis=-1)
    _tcca_scores(writhe,
                 lags,
                 20,
                 path=f"{writhe_path}/lengths_{'_'.join(map(str, [i+1 for i in pair]))}")

