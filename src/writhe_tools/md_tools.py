from .utils.misc import to_numpy,optional_import
md = optional_import("mdtraj", "mdtraj")

import numpy as np
import ray
import warnings
import multiprocessing
from .utils.filing import save_dict, load_dict
from .utils.indexing import (triu_flat_indices,
                            combinations,
                            product,)
from .utils.sorting import filter_strs, lsdir
from .plots import plot_distance_matrix, build_grid_plot
from numba import njit, prange
import os







@njit
def norm(a):
    return np.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:,1] +  a[:, 2] * a[:, 2])

@njit
def distance_pair(xyz, pair, length=None):
    displacements = xyz[:,pair[1]] - xyz[:,pair[0]]
    # if we have periodic boundary conditions
    if length is not None:
        displacements -= length * np.round(displacements / length)
    return norm(displacements) # return only the Euclidean distance


@njit
def parallel_distances(xyz, pairs, length=None):
    distances = np.zeros((len(pairs),len(xyz)))
    for i in prange(len(pairs)):

        distances[i] = distance_pair(xyz, pairs[i], length)
    return distances

def to_distance_matrix(distances: np.ndarray,
                       n: int,
                       m: int = None,
                       d0: int = 1):
    distances = distances.squeeze()
    assert distances.ndim < 3, "distances must be either 1 or two dimensional"
    distances = distances.reshape(1, -1) if distances.ndim < 2 else distances

    # info about flattened distance matrix
    N, d = distances.shape

    # intra molecular distances
    if m is None:
        matrix = np.zeros([N] + [n] * 2)
        i, j = np.triu_indices(n, d0)
        matrix[:, i, j] = distances
        return (matrix + matrix.transpose(0, 2, 1)).squeeze()

    else:
        assert d == n * m, \
            "Given dimensions (n,m) do not correspond to the dimension of the flattened distances"

        return distances.reshape(-1, n, m).squeeze()


def residue_distances(traj,
                      index_0: np.ndarray,
                      index_1: np.ndarray = None,
                      periodic: bool = True,
                      parallel: bool = False):
    # intra distance case
    if index_1 is None:
        indices = combinations(index_0)
        return md.compute_contacts(traj, indices, periodic=periodic)[0], indices if not parallel \
            else parallel_distances(traj.xyz, indices, length=traj.unitcell_lengths if periodic else None)

    # inter distance case
    else:
        indices = product(index_0, index_1)
        return md.compute_contacts(traj, indices, periodic=periodic)[0], indices if not parallel \
            else parallel_distances(traj.xyz, indices, length=traj.unitcell_lengths if periodic else None)


def to_contacts(distances: np.ndarray, cut_off: float = 0.5):
    return np.where(distances < cut_off, 1, 0)




class ResidueDistances:
    def __init__(self,
                 index_0: np.ndarray = None,
                 traj=None,
                 index_1: np.ndarray = None,
                 chain_id_0: str = None,
                 chain_id_1: str = None,
                 contact_cutoff:float=0.5,
                 parallel: bool = False,
                 args: "dict or path (str) to saved dict" = None):

        # restore the class from dictionary that it saves
        if args is not None:
            if isinstance(args, str):
                args = load_dict(args)
            assert isinstance(args, dict), "args must be dict or path to saved dict"
            self.__dict__.update(args)

        # set up new instance, compute distances
        else:
            assert all(arg is not None for arg in (index_0, traj)), (
                "Minimum inputs are MDTraj object and index_0 (residue indices array)"
                " if an args dict saved by this class is not provided")
            self.contact_cutoff = contact_cutoff

            if index_1 is None:
                self.chain_id_0, self.chain_id_1 = [chain_id_0] * 2
                self.residues_0, self.residues_1 = [get_residues(traj, index_0)] * 2
                self.n = len(index_0)
                self.m = None
                self.prefix = "Intra"
                self.distances = residue_distances(traj, index_0,
                                                   parallel=parallel)[0]

            else:
                self.chain_id_0, self.chain_id_1 = chain_id_0, chain_id_1
                self.residues_0, self.residues_1 = get_residues(traj, [index_0, index_1])
                self.n, self.m = map(len, (index_0, index_1))
                self.prefix = "Inter"
                self.distances = residue_distances(traj, index_0, index_1,
                                                   parallel=parallel)[0]

        pass

    def save(self,
             path: str = None,
             dscr: str = None,
             ) -> None:

        if path is not None:
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            path = os.getcwd()

        file = (f"{path}/distance_dict" if dscr is None
                else f"{path}/{dscr}_distance_dict") + ".pkl"

        args = {attr: getattr(self, attr) for attr in filter(lambda x: hasattr(self, x),
                                                             ["distances", "index_0", "index_1",
                                                              "chain_id_0", "chain_id_1", "residues_0",
                                                              "residues_1", "n", "m", "prefix",
                                                              "contact_cutoff"])}
        save_dict(file, args)

        return None

    @classmethod
    def load(cls, file):
        return cls(args=load_dict(file))

    def sub_diag(self, d):
        assert self.prefix == "Intra", "Must be intra molecular distances to subsample distances"
        return self.distances[:, triu_flat_indices(self.n, 1, d)]

    def matrix(self,
               distances: np.ndarray=None,
               contacts: bool = False,
               cut_off: float = None,
               ):

        distances = self.distances if distances is None else distances
        return to_distance_matrix(to_contacts(distances, cut_off or self.contact_cutoff)\
                                  if contacts else distances,
                                  self.n,
                                  self.m)

    def contacts(self, cut_off: float=None):
        cut_off = cut_off if cut_off is not None else self.contact_cutoff
        return to_contacts(self.distances, cut_off=cut_off)

    def plot(self,
             index: "list, int, str" = None,
             contacts: bool = False,
             contact_cutoff: float = None,
             dscr: str = "",
             label_stride: int = 3,
             font_scale: int = 1,
             cmap: str = "jet",
             line_plot_args: dict = None,
             xticks_rotation: float = 0,
             ax=None):



        if contacts:
            matrix = self.contacts(contact_cutoff)
            __dtype = "Contacts"
            unit = ""
        else:
            matrix = self.distances
            __dtype = "Distances"
            unit = " (nm)"

        if index is None:
            matrix = self.matrix(matrix.mean(0))
            __stype = "Average "

        elif isinstance(index, (int, float)):
            index = int(index)
            matrix = self.matrix(matrix[index])
            __stype = ""
            dscr = f"Frame {index}"

        else:
            index = to_numpy(index).astype(int)
            matrix = self.matrix(matrix[index].mean(0))
            __stype = "Average "

            if dscr == "":
                warnings.warn(("Taking the average over a subset of indices."
                               "The option, 'dscr', (type:str) should be set to provide \
                                a description of the indices. "
                               "Otherwise, the plotted data is ambiguous"))

        if dscr != "":
            dscr = f" : {dscr}"

        title = f"{__stype}{self.prefix}molecular {__dtype}{dscr}"
        cbar_label = f"{__stype}{__dtype}{unit}"

        matrix_plot_args = {"matrix": matrix,
                            "ylabel": self.chain_id_0,
                            "yticks": self.residues_0,
                            "xlabel": self.chain_id_1,
                            "xticks_rotation": xticks_rotation,
                            "xticks": self.residues_1,
                            "title": title,
                            "cbar_label": cbar_label,
                            "label_stride": label_stride,
                            "font_scale": font_scale,
                            "ax": ax,
                            "cmap": cmap}

        if line_plot_args is None:
            return plot_distance_matrix(**matrix_plot_args)

        else:
            line_plot_args["xticks"] = self.residues_1
            line_plot_args["x"] = np.arange(len(self.residues_1))
            line_plot_args["font_scale"] = 1
            line_plot_args["hide_title"] = True

            matrix_plot_args["font_scale"] = 1.2
            matrix_plot_args["hide_x"] = True
            matrix_plot_args["xlabel"] = None
            matrix_plot_args["xticks"] = None
            matrix_plot_args["aspect"] = "auto"

            build_grid_plot(matrix_plot_args, line_plot_args)

            return

def rmsd_sort(indices: np.ndarray,
              traj: md.Trajectory,
              target_index: int = 0,
              target_structure: md.Trajectory = None):

    target = target_structure if target_structure is not None else traj[indices[target_index]]
    return indices[md.rmsd(traj[indices], target).argsort()]


def load_traj(dir: str,
              traj_keyword: str = None,
              pdb_keyword: str = None,
              keyword: str = None,
              exclude: str = None,
              selection: str = None,
              traj_ext: str = "dcd",
              top_ext: str = "pdb",
              stride: int = 1):
    def check(x: list):
        if len(x) == 1:
            return x[0]
        else:
            raise Exception("Keyword search returned multiple files, cannot load without ambiguity")

    extensions = [traj_ext, top_ext]

    files = lsdir(dir, keyword=extensions, exclude=exclude, match=any)

    if all(i is not None and isinstance(i, str) for i in (traj_keyword, pdb_keyword)):

        dcd, pdb = [check(filter_strs(files, [ext, kw], match=all))
                    for ext, kw in zip(extensions, [traj_keyword, pdb_keyword])]

    else:

        if keyword is None:
            dcd, pdb = [check(filter_strs(files, keyword=ext, match=all))
                        for ext in extensions]
        else:

            assert isinstance(keyword, (list, str)), "keyword must be list or str"

            keyword = [keyword] if isinstance(keyword, str) else keyword

            print(keyword)

            dcd, pdb = [check(filter_strs(files, [ext] + keyword, match=all))
                        for ext in extensions]

    indices = None if selection is None else md.load(pdb).top.select(selection)

    print(f"Loading Trajectory File : {dcd} with top file : {pdb}")

    return md.load(dcd, top=pdb, atom_indices=indices, stride=stride), dcd, pdb


def calc_sa(trj: "str to traj file or mdtraj Trajectory object",
            helix: "str to helix pdb file or mdtraj object",
            ):

    trj, helix = (traj_slice(j, "name CA").center_coordinates() for j in
                 (md.load(i) if isinstance(i, str) else i for i in (trj, helix)))

    assert trj.n_atoms == helix.n_atoms, \
        "trj and helix trajectories do not contain the same number of CA atoms"

    selections = [f"resid {i} to {i + 5}" for i in range(0, trj.n_atoms - 5)]

    rmsd = np.asarray([md.rmsd(trj, helix, atom_indices=helix.topology.select(selection))
                       for selection in selections])

    sa = (1.0 - (rmsd / 0.08) ** 8) / (1 - (rmsd / 0.08) ** 12)

    return sa.T


residue_volumes = dict(zip(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K',
                            'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
                           [121.0, 265.0, 187.0, 187.0, 148.0, 214.0, 214.0, 97.0,
                            216.0, 195.0, 191.0, 230.0, 203.0, 228.0, 154.0, 143.0,
                            163.0, 264.0, 255.0, 165.0]))


def calc_rsa(traj: "traj object or str",
             pdb: str = None,
             file_name: str = None,
             parallel=False,
             cpus_per_job: int = 1,
             ca_indices: np.ndarray = None):
    if isinstance(traj, str):
        assert pdb is not None, "If traj is a file, must pass pdb file to load traj"
        traj = md.load(traj, top=pdb)

    # helper functions to avoid making copies of the traj and
    # implement iterations smoothly
    traj_slice_ = lambda string: traj.atom_slice(traj.top.select(string))

    # determine protein alpha carbon indices
    if ca_indices is None:
        ca_indices = traj.top.select("protein and name CA")

    # find residue indices and codes of alpha carbons and atoms per residue
    ## this approach avoids selecting ligand atoms and caps
    ### (volume norms only available for the 20 canonical residues)

    index, code, count = np.array([[getattr(traj.top.atom(int(i)).residue, attr)
                                   for attr in ["index", "code", "n_atoms"]]
                                   for i in ca_indices]).T
    count = count.astype(int)

    # make selection for residues of CA atoms
    selection = " or ".join(map("resid {}".format, index))

    # find normalization for particular sequence
    norm = np.vectorize(residue_volumes.__getitem__)(code)

    # compute sasa over all atoms of CA residues
    if parallel:

        # start ray instance
        ray.init()

        traj_ref = ray.put(traj_slice_(selection))

        fxn = ray.remote(lambda traj, index: md.shrake_rupley(traj[index]))

        # chunk indices
        chunks = np.array_split(np.arange(traj.n_frames),
                                int(multiprocessing.cpu_count() / cpus_per_job))
        # run parallelized code
        sasa = np.concatenate(ray.get([fxn.remote(traj=traj_ref, index=index)
                                       for index in chunks]))


        # free memory, shutdown ray instance
        ray.internal.free(traj_ref)
        ray.shutdown()

    else:
        sasa = md.shrake_rupley(traj_slice_(selection))

    # MDTraj computes sasa by atom, we have to sum over residues
    # MDTraj can organize by residues, but we don't trust MDTraj to get the right atoms
    split = np.cumsum(count)[:-1]
    residue_sasa = np.stack([s.sum(1) for s in np.array_split(sasa, split, 1)], 1)

    # normalize and account for angstrom to nm conversion
    rsa = 100 * residue_sasa / norm

    # save data
    if file_name is not None:
        np.save(file_name, rsa)

    return rsa


def Rg(x: np.ndarray) -> float:
    """
    Unweighted radius of gyration
    Args
        x : (n_frames, n_atoms, 3) np.ndarray
    """

    return np.power(np.linalg.norm(x - x.mean(-1, keepdims=True), axis=-1), 2).mean(-1) ** (1/2)


def traj_slice(traj, selection):
    return traj.atom_slice(traj.top.select(selection))


def get_residues(traj,
                 indices: "iterable of arrays or array" = None,
                 atoms: bool = False,
                 cat: bool = False):
    indices = np.arange(len(traj.top.select("name CA"))) if indices is None else indices
    assert isinstance(indices, (np.ndarray, list)), "indices must be np.ndarray or list of np.ndarrays"

    func = (lambda index: traj.top.select(f"resid {int(index)}")) if atoms else (
        lambda index: str(traj.top.residue(int(index))))

    if isinstance(indices, np.ndarray):
        return func(indices) if len(indices) == 1 else \
            np.concatenate(list(map(func, indices))) if (cat and atoms) else \
                list(map(func, indices))
    else:
        return [np.concatenate(list(map(func, i))) if (len(i) != 1 and cat and atoms) else
                list(map(func, i)) if len(i) != 1 else func(i) for i in indices]


