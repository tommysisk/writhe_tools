# Code accompanying the paper, 'On the application of knot theoretic descriptions of proteins to dynamics and generative modeling'

<p align="center">
  <img src="./images/writhe_asyn.png" width="400"/>
  <img src="./images/visualize_writhe.png" width="400"/>
</p>


[//]: # (![Alt text]&#40;./images/writhe_asyn.png&#41;)

[//]: # ()
[//]: # (<p align="center" style="font-size:25px;">)

[//]: # (  Computation of the Writhe)

[//]: # (</p>)

[//]: # ()
[//]: # (![Alt text]&#40;./images/visualize_writhe.png&#41;)




The package contains the following ...

- Numerical routines for computing the writhe using CPU or GPU devices. In either case, computations are parallelized over CPU / GPU cores / devices.
- A class architecture for writhe computation and visualization, making implementation and visualization of the results seamless and efficient. 
- An implementation of the novel writhe-based graph attention message passing layer.
- An implementation of the SE3 equivariant, writhe-PaiNN neural netowrk architecture where SE3 equivariance is acheived by only augmenting invariant graph features.
- Implementations of the orginial PaiNN architecture and the cPaiNN achitecture for comparison.
- An implementation of a score based diffusion model to train all architectures.
- Classes to compute (time-lagged) canonical correlation analysis and visualize results.


For an example of how to use this package to analyze molecular dynamics simulation data, see analysis_example.ipynb.





To train a score-based generative model with any of the architectures listed above, see the sbm_scripts folder 

![Alt text](./images/writhe_layer.png)
