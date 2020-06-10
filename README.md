# Distribution Regression for Continuous-Time Processes
Code to perform distribution regression on time-series via two models (kerES and linSES) based on the Expected Signature.

## Structure of the repository

- The `src` folder contains the implementation of kerES and linSES, as well as two baselines (RBF-RBF [[1]](#1) and DeepSets [[2]](#2))

## Acknowledgments

* The code for the DeepSets model is taken from https://github.com/manzilzaheer/DeepSets
* The code to simulate ideal gases is adapted from https://github.com/labay11/ideal-gas-simulation

## References
<a id="1">[1]</a> 
@inproceedings{muandet2012learning,
  title={Learning from distributions via support measure machines},
  author={Muandet, Krikamol and Fukumizu, Kenji and Dinuzzo, Francesco and Sch{\"o}lkopf, Bernhard},
  booktitle={Advances in neural information processing systems},
  pages={10--18},
  year={2012}
}
<a id="2">[2]</a> 
@inproceedings{zaheer2017deep,
  title={Deep sets},
  author={Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Russ R and Smola, Alexander J},
  booktitle={Advances in neural information processing systems},
  pages={3391--3401},
  year={2017}
}
