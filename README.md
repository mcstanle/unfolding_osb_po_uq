# Introduction
This respository is designed to allow for the reproducibility of "Uncertainty quantification for wide-bin unfolding: one-at-a-time strict bounds and prior-optimized confidence intervals" (https://arxiv.org/abs/2111.01091).

This codebase is setup to be run from scratch (i.e., you get to generate everything yourself), or from pre-computed objects (i.e., smearing matrices, data, etc.). It should be noted that almost all the code in this repository can be run on a reasonable local machine. The two parts requiring more computational firepower are the brute-force computation of the adversarial ansatz (`brute_force_data_gen_ansatz.py`) and the parallelized computation of the nine different interval types for the steeply falling spectrum example (`steeply_fallling_spectra_parallel.py`).

# Setting up the python environment
The python requirements for running the code in this repository are stored in `osb_po_uq.yml`, which can be used to replicate the correct computational environment in conda. To create this environment, simply run:
`$ conda env create -f osb_po_uq.yml`.

# Generating Computational Materials
In order to perform the simulation experiments, the smearing matrices, bin means, and functionals have to be generated.

First, here is the general order in which one should run the components of this repository:
1. Deconvolution Example
    1. `compute_K_deconvolution.py`
    2. `compute_bin_means_deconvolution.py`
    3. `compute_K_deconvolution_adversarial_ansatz.py`
2. Steeply Falling Spectrum Example

## Generating Smearing Matrices
The smearing matrices for the deconvolution and steeply falling spectrum examples are generated in `compute_K_deconvolution.py` and `compute_K_steeply_falling_spectrum.py`. Each of these files has switches to toggle the matrices one wishes to compute. By default, the matrices for both experiments are saved in `./smearing_matrices/`, thought this can be changed by altering `MATRIX_BASE_LOC` in each file.

## Generating Bin Means
Text.

## Generating Functionals
Text.