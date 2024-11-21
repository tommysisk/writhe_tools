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
atom_indices = md.load("example.pdb").top.select("name = CA")
trajectory = md.load("example.xtc", top="example.pdb", atom_indices=atom_indices)
xyz = trajectory.center_coordinates().xyz
```

NOTE: it is generally sufficient to compute the writhe using only the coordinates of the alpha carbons. In principle, one could include all backbone atoms
or even more nuanced selections if one is looking to more specific geometric element. In general, however,
we did not find the addition of more backbone atoms particularly useful in our study.


We can now instantiate an instance of the Writhe class.
```jupyterpython
from writhe_tools.writhe import Writhe
writhe = Writhe(xyz=xyz)
```
To compute the writhe at a given segment length, we use the class method, compute_writhe. This method has many options. Here's a pseudo definition of the class method
with descriptions of the arguments. 
```jupyterpython
writhe.compute_writhe(length: "Define segment size : CA[i] to CA[i+length], type : int",
                       matrix: "Return symmetric writhe matrix, type : bool" = False,
                       store_results: "Bind calculation results to class for plotting, type : bool" = True,
                       xyz: "Coordinates to use in writhe calculation (n, points/atoms, 3), type : np.ndarray" = None,
                       n_points: "Number of points in each topology used to estimate segments, type : int " = None,
                       speed_test: "Test the speed of the calculation and return nothing, type : bool" = False,
                       cpus_per_job: "Number of CPUs to allocate to each batch, type : int" = 1,
                       cuda: "Use cuda enabled devices to compute the writhe (will multiprocess if available), type : bool" = False,
                       cuda_batch_size: "Number of segments to compute per batch if using cuda, type : bool" = None,
                       multi_proc: "Use multi_processing in calculation (applies to either CPU or GPU), type : bool" = True,
                       ):->dict
```
The class method, *compute_writhe* is defined so that any set of coordinates can be dropped in to the calculation. However, leaving the argument as default
will use the coordinates (xyz) the class was instantiated with. 

The only required argument is length (int), which defines the segment length (see our paper). The other arguments allow 
specification of the compute strategy. Multiprocessing is supported for both GPU and CPU computation. By default, the 
method performs a multiprocessed CPU computation (multi_proc=True, cuda=False). GPUs (multi_proc=True/False, cuda=True) 
will substantially increase performance. The GPU batch size can be manually set (cuda_batch_size) and should be if GPU memory errors are encountered.
If left as None, a conservative guess of the appropriate batch will be used.

NOTE the class will automatically switch to CPU calculation if cuda is not available.

Below we show how to compute the writhe, save the results and restore the class from the saved results.
```jupyterpython
# compute the writhe using segment length 1 and default arguments
writhe.compute_writhe(length=1)
# save the result with default arguments (None), will make a file in working directory called "./writhe_data_dict_length_1.pkl"
writhe.save(dir=None, dscr=None)
# restore the calculation at a later time
restored_writhe = Writhe("./writhe_data_dict_length_1.pkl")
```
The results are saved as a pickled python dictionary with a pre-defined name:
```jupyterpython
f"{dir}/{dscr}_writhe_data_dict_length_{self.length}.pkl"
```
Where *dir* and *dscr* are optional parameters of the *save* method and can be used to define the directory and a file description
to save the result under, respectively.


The class also has the plotting methods, defined in pseudo code, below: 
```jupyterpython
writhe.plot_writhe_matrix(ave=True, index: "int or list or str" = None,
                           absolute=False, xlabel: str = None, ylabel: str = None,
                           xticks: np.ndarray = None, yticks: np.ndarray = None,
                           label_stride: int = 5, dscr: str = None,
                           font_scale: float = 1, ax=None)->matplotlib.pyplot.figure:


```





