## Code accompanying the paper, 'On the application of knot theoretic geometric descriptors to dynamical and generative models'


<p align="center" style="font-size:40px;">
  Computation of the writhe and analysis of polymer coordinate data
</p>


<p align="center">
  <img src="./images/visualize_writhe.png" width="410"/>
  <img src="./images/writhe_asyn.png" width="410"/>
</p>





The package contains the following ...

- Numerical routines for computing the writhe using CPU or GPU devices. In either case, computations are (optionally) parallelized over CPU / GPU cores / devices.
- A class architecture for writhe computation and visualization. 
- An implementation of the novel writhe-based graph attention message passing layer.
- An implementation of the SE3 equivariant, writhe-PaiNN neural network architecture where SE3 equivariance is acheived by only augmenting invariant graph features.
- Implementations of the orginial PaiNN architecture and the cPaiNN achitecture for comparison.
- An implementation of a score based diffusion model to train all architectures.
- Classes to compute (time-lagged) canonical correlation analysis and visualize results.


#### For an example of how to use this package to analyze molecular dynamics simulation data, see analysis_example.ipynb in the examples' folder and the mini tutorial below





#### To train score-based generative models with any of the architectures listed above, see the scripts folder.



![Alt text](./images/writhe_layer.png)

---

## The main tool in this package is the class:

```jupyterpython
writhe_tools.writhe.Writhe
```

This class is instantiated with one argument, xyz, which should be an (N samples, D points or atoms, 3 coordinates) numpy array.
For a molecular dynamics trajectory and structure file, the required input can be obtained as shown below.  

Here, we use MDTraj to load the trajectory. 




```jupyterpython
import mdtraj as md 
atom_indices = 
xyz = md.load("example.xtc", top="example.pdb",
              atom_indices=md.load("example.pdb").top.select("name = CA")
              ).center_coordinates().xyz

```

NOTE: it is generally sufficient to compute the writhe using only the coordinates of the alpha carbons. In principle, one could include all backbone atoms
or any selection of contiguous atoms or points.

We can now instantiate an instance of the Writhe class.
```jupyterpython
from writhe_tools.writhe import Writhe
writhe = Writhe(xyz=xyz)
```

We can then compute the writhe at a given segment length, save the result for later and then restore the class
from the saved result to continue analysis or visualization.

```jupyterpython
# compute the writhe using segment length 1 and default arguments
writhe.compute_writhe(length=1)

# save the result with default arguments (None), 
# will make a file in working directory called "./writhe_data_dict_length_1.pkl"

writhe.save(path=None, dscr=None)

# restore the calculation at a later time using the CLASS method, load
restored_writhe = Writhe.load("./writhe_data_dict_length_1.pkl")
```

The results are saved as a pickled python dictionary with a pre-defined name:
```jupyterpython
f"{path}/{dscr}_writhe_data_dict_length_{self.length}.pkl"
```
Or if path and dscr are left to None:
```jupyterpython
f"./writhe_data_dict_length_{self.length}.pkl"
```
 The *compute_writhe* method has many options. Here's an example with descriptions of all the arguments.
 Only the argument defining the segment length, **length**, is required. Note that calculation can be performed 
on multiple (multi_proc=True) CPU or GPU (cuda=True) devices. The method defaults to a multiprocessed CPU calculation
but can be greatly expedited with GPUs. If using GPUs, it is best to avoid interactive kernels like jupyter notebooks due 
to known issues with clearing GPU memory. One may also have to manually set cuda_batch_size to avoid out of
memory errors. 

```jupyterpython
results = writhe.compute_writhe(
          length=1,  # Default segment length
          matrix=False,  # Default: Do not return the symmetric writhe matrix
          store_results=True,  # Default: Bind calculation results to class for plotting
          xyz=None,  # Default: Use the coordinates from the class instance (self.xyz)
          n_points=None,  # Default: Use n_points from the class instance (self.n_points)
          speed_test=False,  # Default: Do not perform speed test
          cpus_per_job=1,  # Default: Use 1 CPU per job
          cuda=False,  # Default: Do not use CUDA (use CPU instead)
          cuda_batch_size=None,  # Default: No batch size for CUDA (not used since cuda=False)
          multi_proc=True  # Default: Use multi-processing
        )
```
The class method, *compute_writhe* is defined so that any set of coordinates (**xyz**, **n_points**)
can be dropped in to the calculation.
However, leaving the argument as default
will use the coordinates (xyz) the class was instantiated with. 


Below we show how to compute the writhe, save the results and restore the class from the saved results.

- NOTES:
  - The *compute_writhe* method returns a dictionary of the results. If store_result=True, there's no need to assign a variable to the return.

  - the argument, **store_result**, of *compute_writhe* must be set to True (Default) in order to plot or save calculation
  results. 

  - The class will automatically switch to CPU calculation if cuda is not available.



The class also has plotting methods with many options
```jupyterpython
writhe.plot_writhe_matrix(
    ave=True,                     # ave: bool = True
                                 # (Averages the writhe matrix across frames by default)
    index=None,                   # index: Optional[Union[int, List[int], str, np.ndarray]] = None
                                 # (Plots the average writhe matrix if index is None)
    absolute=False,               # absolute: bool = False
                                 # (Uses signed writhe values by default)
    xlabel=None,                  # xlabel: Optional[str] = None
                                 # (No custom label for the x-axis, default will be used)
    ylabel=None,                  # ylabel: Optional[str] = None
                                 # (No custom label for the y-axis, default will be used)
    xticks=None,                  # xticks: Optional[np.ndarray] = None
                                 # (No custom xticks provided, default will be used)
    yticks=None,                  # yticks: Optional[np.ndarray] = None
                                 # (No custom yticks provided, default will be used)
    label_stride=5,               # label_stride: int = 5
                                 # (Tick labels will be spaced every 5 units by default)
    dscr=None,                    # dscr: Optional[str] = None
                                 # (No description for the subset of frames averaged)
    font_scale=1,                 # font_scale: float = 1
                                 # (Font size will be at the default scale)
    ax=None                       # ax: Optional[plt.Axes] = None
                                 # (No custom Axes object provided, so a new figure will be created)
)

writhe.plot_writhe_per_segment(
    ave=True,                      # ave: bool = True
                                  # (Averages over all frames by default)
    index=None,                    # index: Optional[Union[int, List[int], str, np.ndarray]] = None
                                  # (Plots the average writhe per segment if index is None)
    xticks=None,                   # xticks: Optional[List[str]] = None
                                  # (No custom xticks are provided; default range is used)
    label_stride=5,                # label_stride: int = 5
                                  # (Tick labels are spaced every 5 segments by default)
    dscr=None,                     # dscr: Optional[str] = None
                                  # (No description for the averaged indices)
    ax=None                        # ax: Optional[plt.Axes] = None
                                  # (No custom Axes object provided; a new figure will be created)
)

self.plot_writhe_total(
    window=None,           # window: Optional[int] = None
                           # (No window averaging applied; raw data is used)
    ax=None                # ax: Optional[plt.Axes] = None
                           # (No custom Axes object provided; a new figure will be created)
)

```





