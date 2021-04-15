# Distribution Regression for Sequential Data
Code to perform distribution regression (DR) on time-series via two models (KES and SES) based on the Expected Signature.
[Link to the paper](https://arxiv.org/pdf/2006.05805.pdf)

<p align="center">
<img src="https://user-images.githubusercontent.com/26120929/114847969-28eb3f00-9dd6-11eb-99a9-34f584ca3b48.gif">
</p>

## Structure of the repository

- The `src` folder contains the implementation of KES and SES for DR on time-series , as well as an implementation of Support Distribution Machines (SDM) using RBF and Matern32 kernels [[1]](#1) for DR on vectorial data. It also includes an implementation of SDM using the GA kernel [[2]](#2) for time-series.
- The `examples` folder contains notebooks to reproduce the experiments of the paper. 
- The `data` folder contains precomputed datasets for the experiments. 

## Dependencies

Python libraries required to run the code can be installed by `pip install -r requirements.txt`. 
- `torch==1.3.0` is only required to train the DeepSets models.
- `scikit_learn==0.23.1` is for the implementation of SES and KES. 
- `esig==0.7.1` is only required to associate names to the features in the SES model. 
- `iisignature==0.24` is the Python library used to compute *signatures* in SES. 
- `fbm==0.3.0` is used to generate fractional Brownian motion samples.

## Acknowledgments

* The code for the DeepSets models is taken from https://github.com/manzilzaheer/DeepSets
* The code to simulate ideal gases is adapted from https://github.com/labay11/ideal-gas-simulation

## References
<a id="1">[1]</a> 
Muandet, Krikamol, et al. "Learning from distributions via support measure machines." Advances in neural information processing systems. 2012.

<a id="2">[2]</a> 
Cuturi, Marco, et al. "A kernel for time series based on global alignments." IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07. 2007.

<a id="3">[3]</a> 
Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems. 2017.

