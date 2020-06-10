# Distribution Regression for Continuous-Time Processes
Code to perform distribution regression on time-series via two models (kerES and linSES) based on the Expected Signature.

## Structure of the repository

- The `src` folder contains the implementation of kerES and linSES, as well as two baselines (RBF-RBF [[1]](#1) and DeepSets [[2]](#2))

## Acknowledgments

* The code for the DeepSets model is taken from https://github.com/manzilzaheer/DeepSets
* The code to simulate ideal gases is adapted from https://github.com/labay11/ideal-gas-simulation

## References
<a id="1">[1]</a> 
Muandet, Krikamol, et al. "Learning from distributions via support measure machines." Advances in neural information processing systems. 2012.
<a id="2">[2]</a> 
Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems. 2017.

