# Introduction
This respository is designed to allow for the reproducibility of ``Uncertainty quantification for wide-bin unfolding: one-at-a-time strict bounds and prior-optimized confidence intervals" (https://arxiv.org/abs/2111.01091).

This codebase is setup to be run from scratch (i.e., you get to generate everything yourself), or from pre-computed objects (i.e., smearing matrices, data, etc.).

# Generating Smearing Matrices
The smearing matrices for the deconvolution and steeply falling spectrum examples are generated in `compute_K_deconvolution.py` and `compute_K_steeply_falling_spectrum.py`. Each of these files has switches to toggle the matrices one wishes to compute. By default, the matrices for both experiments are saved in `./smearing_matrices/`, thought this can be changed by altering `MATRIX_BASE_LOC` in each file.