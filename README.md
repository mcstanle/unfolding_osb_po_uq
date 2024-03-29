# Introduction
This respository is designed to allow for the reproducibility of "Uncertainty quantification for wide-bin unfolding: one-at-a-time strict bounds and prior-optimized confidence intervals" (https://doi.org/10.1088/1748-0221/17/10/P10013).

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
    1. `compute_K_steeply_falling_spectrum.py`
    2. `compute_bin_means_steeply_falling_spectrum.py `

## Generating Smearing Matrices
The smearing matrices for the deconvolution and steeply falling spectrum examples are generated in `compute_K_deconvolution.py` and `compute_K_steeply_falling_spectrum.py`. Each of these files has switches to toggle the matrices one wishes to compute. By default, the matrices for both experiments are saved in `./smearing_matrices/`, thought this can be changed by altering `MATRIX_BASE_LOC` in each file.

## Generating Bin Means
Bin means for the deconvolution and steeply falling spectrum examples are computed in `compute_bin_means_deconvolution.py` and `compute_K_steeply_falling_spectrum.py`. For the deconvolution example, this script handles the true and smear bin means for three setups; wide bin (10 true/40 smear), full-rank (40 true/40 smear), and rank-deficient (80 true/40 smear).

## Adversarial Ansatz, Matrices, and Bin Means
These are all computed with the script, `compute_K_deconvolution_adversarial_ansatz.py`. This script relies upon `./data/brute_force_ansatz/ansatz_data_gmm.npy` and `./data/brute_force_ansatz/coverages_gmm.npy` to compute the adversarial ansatz. The first data set is a collection of 1000 bin-count realizations from the true poisson process and can be reproduced by running `data_generation_ansatz_data.py`. To reproduce the paper results exactly, please use the original data in the repository. The second dataset is computed in `brute_force_data_gen_ansatz.py`. This script relies upon parallelization and was run on CMU Stat's compute cluster. While in principle it can be run on a local machine, it has not been tested in that context. We recommend, again, using the generated data set.

## Generating Functionals
All bin aggregation functionals are generated in `create_functionals.py`. Both sets of functionals are saved in `./functionals/`.

## Data Generation
To empirically evaluate the coverage of these interval estimator, we must sample many times from the true distribution to fit each interval method many times. Hence, we must create ensembles of true data.

### Wide-Bin Deconvolution
The original data used for the analysis in the paper is included in `./data/wide_bin_deconvolution/simulation_data_ORIGINAL.npy`. However, the code used to generate that data is found in `data_generation_deconvolution.py`. It should be noted that if one wants to generate their own data using this file, results will differ slightly from those in the paper.

### Steeply Falling Particle Spectrum
The original data used for the analysis in the paper is included in `./data/steeply_falling_spectrum/data_ORIGINAL.npy`. If one wishes to generate the data again for oneself, please use the script `data_generation_steeply_falling_spectrum.py`. As above, it should be noted that these data will differ from the original, leading to slightly different results.

Additionally, since this example includes shape constraints, see `generate_shape_constraints.py` for the code to generate the matrix shape constraints. This script has been pre-run, resulting in the file `./data/steeply_falling_spectrum/constraint_matrices.npz`.

# Code to fit the different types of intervals
We fit many different interval estimators throughout the paper. All of these computational methods can be found in `interval_estimators.py`. This script contains functions for fitting the folling intervals:
1. Least-squares
2. OSB
3. PO
4. SSB
5. Fixed-width Minimax Bounds

# Results
For both sets of results, we store all plot code in a separate file from the script that actually creates the plots (see each sub-section below for details).

## Application to Wide-Bin Deconvolution
Plotting functions are stored in `plotting_deconvolution.py`. Plots can be generated by running `plot_create_deconvolution.py`. For each setup, a file of generated intervals and coverages is generated. These are stored in `./data/wide_bin_deconvolution/`.

In the paper, we step through the following sequence of results:
1. Unfolding using least-squares intervals (`deconvolution_wide_bins_least_squares.py`)
    1. Directly to wide-bins (dramatic undercoverage) (`./data/wide_bin_deconvolution/ints_cov_wide_ls.npz`)
    2. Directly to fine-bins (addressed undercoverage) (`./data/wide_bin_deconvolution/ints_cov_fine_ls.npz`)
    3. Unfolding to fine-bins with post-inversion aggregation (addressed undercoverage) (`./data/wide_bin_deconvolution/ints_cov_agg_ls.npz`)
    4. Unfolding with OSB and PO intervals with the GMM Ansatz (`./deconvolution_wide_bins_osb_po.py`)
    5. Comparing OSB and PO with SSB and Minimax (`./deconvolution_wide_bins_ssb_minimax.py`)
    6. Effects of number of true bins and prior choice on the expected width of the PO intervals
2. Wide-bin unfolding with OSB and PO intervals (`deconvolution_wide_bins_least_squares.py`)
    1. GMM Ansatz (minor misspecification) (`./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz.npz`)
    2. Adversarial Ansatz (substantial misspecification)

## Application to Unfolding a Steeply Falling Particle Spectrum
Plotting functions are stored in `plotting_steeply_falling_spectrum.py`. Plots can be generated by running `plot_create_steeply_falling_spectrum.py`. For each setup, a file of generated intervals and coverages is generated. These are stored in `./data/steeply_falling_spectrum/`.

Since computing the intervals across OSB/PO/SSB, the three possible constraint matrices, and the ten functionals is a heavy computation and embarrassingly parallelizable, the results of those computations used in the paper are stored in `./data/steeply_falling_spectrum/intervals_optimized_ansatz_rank_def_uneven_bin.npy`. The results were stored in a 5-dimensional array indexed as follows; first index denotes interval type, second index denotes constraint type, third index denotes functional number, fourth index provides the interval lower bound, and the fifth index provides the upper bound. This indexing is used when accessing these intervals in `plot_figure18` in `plotting_steeply_falling_spectrum.py`, for example.