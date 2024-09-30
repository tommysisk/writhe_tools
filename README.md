## Code accompanying the paper, 'On the application of knot theoretic descriptions of proteins to dynamics and generative modeling'


<p align="center" style="font-size:40px;">
  Computation of the writhe and analysis of polymer coordinate data
</p>


<p align="center">
  <img src="./images/visualize_writhe.png" width="410"/>
  <img src="./images/writhe_asyn.png" width="410"/>
</p>





The package contains the following ...

- Numerical routines for computing the writhe using CPU or GPU devices. In either case, computations are parallelized over CPU / GPU cores / devices.
- A class architecture for writhe computation and visualization, making implementation and visualization of the results seamless and efficient. 
- An implementation of the novel writhe-based graph attention message passing layer.
- An implementation of the SE3 equivariant, writhe-PaiNN neural netowrk architecture where SE3 equivariance is acheived by only augmenting invariant graph features.
- Implementations of the orginial PaiNN architecture and the cPaiNN achitecture for comparison.
- An implementation of a score based diffusion model to train all architectures.
- Classes to compute (time-lagged) canonical correlation analysis and visualize results.


#### For an example of how to use this package to analyze molecular dynamics simulation data, see analysis_example.ipynb.





#### To train score-based generative models with any of the architectures listed above, see the sbm_scripts folder 

![Alt text](./images/writhe_layer.png)
