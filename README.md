# Distribution Regression for Continuous-Time Processes
Code to perform distribution regression (DR) on time-series via two models (kerES and linSES) based on the Expected Signature.

## Structure of the repository

- The `src` folder contains the implementation of kerES and linSES for DR on time-series , as well as an implementation of Support Distribution Machines (SDM) using RBF kernels [[1]](#1) for DR on vectorial data. 
- The `examples` folder contains notebooks to reproduce the experiments of the paper. 
- The `data` folder contains precomputed datasets for the experiments. 

## Acknowledgments

* The code for the DeepSets model is taken from https://github.com/manzilzaheer/DeepSets
* The code to simulate ideal gases is adapted from https://github.com/labay11/ideal-gas-simulation

## References
<a id="1">[1]</a> 
Muandet, Krikamol, et al. "Learning from distributions via support measure machines." Advances in neural information processing systems. 2012.

<a id="2">[2]</a> 
Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems. 2017.

