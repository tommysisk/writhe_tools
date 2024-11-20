import numpy as np
import torch
import mdtraj
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import InMemoryDataset
from .indexing import product


class GraphDataSet(InMemoryDataset):
    def __init__(self, data_list=None, file: str = "graphs.pt"):
        super().__init__()
        if data_list is not None:
            data, slices = self.collate(data_list)
            torch.save((data, slices), file)

        self.data, self.slices = torch.load(file)


def dict_map(dic, keys):
    check = list(dic.keys())
    assert all(k in check for k in keys), "Not all keys exist in dict"
    return list(map(dic.__getitem__, keys))


single_letter_codes = ["G", "A", "S", "P", "V", "T", "C", "L", "I", "N",
                       "D", "Q", "K", "E", "M", "H", "F", "R", "Y", "W"]

three_letter_codes = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "LEU", "ILE", "ASN",
                      "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]

abr_to_code_ = dict(zip(three_letter_codes, single_letter_codes))

code_to_index_ = dict(zip(single_letter_codes, range(len(single_letter_codes))))

bond_codes_ = torch.zeros(20, 20).long()
bond_code_values_ = torch.arange(1, 211)
i_, j_ = torch.triu_indices(20, 20, 0)
bond_codes_[i_, j_] = bond_code_values_
i_, j_ = torch.triu_indices(20, 20, 1)
bond_codes_[j_, i_] = bond_codes_[i_, j_]


def get_codes(traj):
    return list(map(str, list(traj.top.to_fasta())[0]))


def abr_to_code(keys):
    return dict_map(abr_to_code_, keys)


def code_to_index(codes):
    if len(codes[0]) > 1:
        codes = abr_to_code(codes)
    return torch.LongTensor(dict_map(code_to_index_, codes))


def get_edges_bonds(index_sequence):
    index_sequence = index_sequence.flatten() if index_sequence.ndim != 1 else index_sequence
    n = len(index_sequence)
    edges = product(np.arange(n), np.arange(n))
    edges = torch.LongTensor(edges[edges[:, 0] != edges[:, 1]]).T

    # bonding info
    where = abs((edges * torch.LongTensor([1, -1]).reshape(2, 1)).sum(0)) == 1
    i, j = edges[:, where]
    values = bond_codes_[index_sequence[i], index_sequence[j]]
    bonds = torch.zeros(edges.shape[-1]).long()
    bonds[where] = values
    return edges, bonds


def make_dataset(traj: mdtraj.Trajectory, file: str = "graphs.pt"):
    traj = traj.atom_slice(traj.top.select("name CA")).center_coordinates()

    index_sequence = code_to_index(get_codes(traj))

    edge_index, bonds = get_edges_bonds(index_sequence)

    xyz = traj.xyz

    scale = np.linalg.norm(xyz.reshape(-1, 3), axis=-1).std()

    xyz /= scale

    print(scale)

    # make data objects
    data_objs = [GeometricData(x=torch.Tensor(x),
                               atoms=index_sequence,
                               edge_index=edge_index,
                               bonds=bonds,
                               )
                 for x in xyz]

    return GraphDataSet(data_list=data_objs, file=file)
