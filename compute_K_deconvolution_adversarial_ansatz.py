"""
Computes the smearing matrix for the adversarial ansatz setup.

This script contains code to do the following
1. generate ansatz data, i.e., the data generated to fit each proposed
   adversarial intensity function.
2. compute the constrained least-squares solution
3. compute the cubic spline interpolator
4. compute the adversarial ansatz smearing matrix

NOTE: this script assumes that the bin-wise coverage for each proposed
adversarial ansatz has been computed in parallel in
./brute_force_data_gen_ansatz.py. That script creates two data files:
1. ansatz_data_gmm.npy
    - data drawn to create each proposed adversarial ansatz
2. coverages_gmm.npy
    - the bin-wise coverage for each proposed adversarial ansatz

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 03 Nov 2021
===============================================================================
"""
import cvxpy as cp
import numpy as np
from scipy import interpolate
from utils import compute_even_space_bin_edges

def identify_min_min_coverage(computed_coverages):
    """
    For each proposed adversarial ansatz, finds the minimum coverage across
    bins. Then, finds the adversarial ansatz with the smallest minimum.

    Parameters:
    -----------
        computed_coverages (np arr) : adversarial ansatz coverages

    Returns:
    --------
        min_min_idx (int) : the index of the proposed adversarial ansatz with
                            smallest bin-wise coverage.
    """
    # find min coverage on each row
    min_row_coverage = computed_coverages.min(axis=1)

    # find the min min
    min_min_idx = np.argmin(min_row_coverage)

    return min_min_idx


def constrained_ls_estimator_gmm_only(data, K, smear_means):
    """
    Fits least squared solution with positivity constraint on the
    parameters;

    Parameters:
    -----------
        data        (np arr) : sampled data to create ansatz
        K           (np arr) : smearing matrix
        smear_means (np arr) : true means to use for the cholesky transformation

    Returns:
    --------
        x_opt.value (np arr) : constrained ls estimate solved by cvxpy
    """
    # compute cholesky transformed K matrix from data
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)
    K_tilde = L_data_inv @ K

    # transform the data
    y = L_data_inv @ data

    # perform the optimization
    x_opt = cp.Variable(K.shape[1])
    ls_constr_prob = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - K_tilde @ x_opt)),
        constraints=[
            x_opt >= 0
        ]
    )
    x_opt_sol = ls_constr_prob.solve()

    # check convergence
    ls_constr_prob.status == 'optimal'

    return x_opt.value


def fit_interpolator_intensity(true_edges, ls_est_vals):
    """
    Fit the intensity function based on an intepolation of the least squares
    estimator.

    Interpolates with a cubic spline from scipy.

    NOTE: this version ensures that the intensity function is always positive.

    Parameters:
    -----------
        true_edges  (np arr) : edges of true bins
        ls_est_vals (np arr) : constrained least squares estimate

    Returns:
    --------
        interp_ansatz_intensity (func) : float input function of intensity function
    """
    # find the width for intensity scaling
    width = true_edges[1] - true_edges[0]

    # interpolate the above
    interp_x_vals = (true_edges[:-1] + true_edges[1:])/2
    interp_ansatz = interpolate.CubicSpline(
        x=interp_x_vals,
        y=ls_est_vals
    )

    interp_ansatz_intensity = lambda x: np.max([interp_ansatz(x) / width, 0])

    return interp_ansatz_intensity


if __name__ == "__main__":

    # base directories
    BASE_DATA_DIR = './data'

    # read in the computed coverages
    computed_coverages = np.load(BASE_DATA_DIR + '/brute_force_ansatz/coverages_gmm.npy')

    # read in the ansatz data
    ansatz_data = np.load(BASE_DATA_DIR + '/brute_force_ansatz/ansatz_data_gmm.npy')

    # import the smearing matrix and means of the true GMM process
    K_obj = np.load(
        file='./smearing_matrices/K_full_rank_mats.npz'
    )
    bin_means_obj = np.load(
        file='./bin_means/gmm_fr.npz'
    )
    K_fr = K_obj['K_fr']
    s_means_fr = bin_means_obj['s_means_fr']

    # find the min-min
    min_min_idx = identify_min_min_coverage(computed_coverages)

    # find the constrained LS solution for the min-min ansatz
    x_opt_min_min = constrained_ls_estimator_gmm_only(
        data=ansatz_data[min_min_idx, :], K=K_fr, smear_means=s_means_fr
    )

    # fit the cubic-spline to the above
    true_edges_fr = compute_even_space_bin_edges(
        bin_lb=-7, bin_ub=7, num_bins=40
    )
    intensity_min_min = fit_interpolator_intensity(
        true_edges=true_edges_fr,
        ls_est_vals=x_opt_min_min
    )