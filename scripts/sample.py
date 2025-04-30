#!/usr/bin/env python
from argparse import ArgumentParser
from sbm import GeometricDDPM
from torch_geometric.loader import DataLoader
from writhe_tools.utils.graph_utils import GraphDataSet
import mdtraj as md
import numpy as np
import os


def get_digit(string):
    out = "".join(filter(str.isdigit, string))
    if len(out) == 0:
        return None
    else:
        return out


def rm_path(string: str):
    return string.split("/")[-1]


def rm_extension(string: str):
    return ".".join(string.split(".")[:-1])


def save_mdtraj(xyz: np.ndarray, file: str, top):
    traj = md.Trajectory(xyz.reshape(-1, top.n_atoms, 3), topology=top)
    traj.save_dcd(get_indexed_file(f"{file}.dcd"))
    return


def get_indexed_file(file):
    """recursive function to create the next iteration in a series of indexed directories
    to save training results generated from repeated trials. Guarantees correctly indexed
    directories and allows for lazy initialization (i.e finds the next index automatically
    assuming the "root" or "base" directory name remains the same)"""

    if not os.path.isfile(file):
        return file
    else:
        path_list = file.split("/")
        local_file = path_list.pop()
        global_dir = "/".join(path_list)+"/"
        no_ext = rm_extension(local_file)
        digit = get_digit(no_ext)

        if digit is not None:
            trial_file = global_dir + local_file.replace(digit, str(int(digit)+1))
        else:
            trial_file = global_dir + no_ext + "1." + local_file.split(".")[-1]

        trial_file = get_indexed_file(trial_file)

        return trial_file


def main(args):
    log_path = "/".join(args.ckpt_file.split("/")[:-1]) if args.log_path is None else args.log_path

    top = md.load(args.template_pdb).top

    model = GeometricDDPM.load_from_checkpoint(checkpoint_path=args.ckpt_file, strict=False)

    dataset = GraphDataSet(file=args.data_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    samples = []
    for i, batch in enumerate(loader):
        samples.append(model.sample_like(batch, ode_steps=args.ode_steps).x.cpu().numpy().reshape(-1, top.n_atoms, 3))
        if i > 182 * 5 * 5: break

    save_mdtraj(xyz=np.concatenate(samples) * args.scale,
                file=f"{log_path}/samples",
                top=top)

    # del samples
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for i, batch in enumerate(loader):
    #     diffusion_traj = model.sample_like(batch,
    #                                        ode_steps=args.ode_steps, save_traj=True)[-1].cpu().numpy().reshape(-1, 20,
    #                                                                                                            3)
    #     save_mdtraj(xyz=diffusion_traj, file=f"{log_path}/diffusion_traj_{i}")
    #
    #     if i > 20: break




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--log_path", required=False, type=str,
                        help="path to saved check point file for resuming training, defaults to None and is not required"
                        )

    parser.add_argument("--ckpt_file", default=None, type=str,
                        help="file to a checkpoint to trained model")

    parser.add_argument("--template_pdb", "-pdb",  type=str, default="asyn_ca.pdb")

    parser.add_argument("--scale", type=float, default=0.34199494)

    main(parser.parse_args())
