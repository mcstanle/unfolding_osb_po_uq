"""
Computes the smearing matrix for the adversarial ansatz setup.

This script contains code to do the following
1. generate ansatz data, i.e., the data generated to fit each proposed
   adversarial intensity function.
2. compute the constrained least-squares solution
3. compute the cubic spline interpolator
4. compute the adversarial ansatz smearing matrix

Generates the smearing matrices and true/smear bin means for both the 40
(full rank) and 80 (rank deficient) true bin cases.

NOTE: this script assumes that the bin-wise coverage for each proposed
adversarial ansatz has been computed in parallel in
./brute_force_data_gen_ansatz.py. That script creates two data files:
1. ansatz_data_gmm.npy
    - data drawn to create each proposed adversarial ansatz
2. coverages_gmm.npy
    - the bin-wise coverage for each proposed adversarial ansatz

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 04 Nov 2021
===============================================================================
"""
import cvxpy as cp
from functools import partial
import json
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from utils import compute_even_space_bin_edges, compute_K_arbitrary

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


def compute_adversarial_bin_means(true_edges, intensity_min_min, K_ansatz):
    """
    Computes the true and smear bin means using the adversarial ansatz.

    Parameters:
    -----------
        true_edges        (np arr) : edges of true bins
        intensity_min_min (func)   : real function for the adversarial ansatz
        K_ansatz          (np arr) : adversarial ansatz smearing matrix

    Returns:
    --------
        ansatz_true_means  (np arr) : true bin means
        ansatz_smear_means (np arr) : smear bin means
    """
    # compute the unfold bin ansatz means
    dim_true = true_edges.shape[0] - 1
    ansatz_true_means = np.zeros(dim_true)

    count = 0
    for i, j in zip(true_edges[:-1], true_edges[1:]):
        ansatz_true_means[count]= quad(intensity_min_min, a=i, b=j)[0]
        count += 1
        
    # compute the smeared means
    ansatz_smear_means = K_ansatz @ ansatz_true_means

    return ansatz_true_means, ansatz_smear_means


def intensity_min_min_const(t, intensity_f, const=10):
    """
    Wrapper around an intensity function that replaces the identically
    zero portions of the domain with a constant. This is to prevent division
    by zero when computing a smearing matrix from an intensity function that
    is identically 0 over some bin.

    Parameters:
    -----------
        t           (float)    : value at which to evaluate the intensity
        intensity_f (function) : the intensity function
        const       (float)    : the constant that replaces 0

    Returns:
    --------
        intensity(f) if < const (float)
    """
    # evaluate intensity_f
    orig_eval = intensity_f(t)
    
    if orig_eval == 0:
        return const
    else:
        return orig_eval


def compute_adversarial_ansatz(
    adv_antz_data, K, s_means, bin_lb, bin_ub, dim_true
):
    """
    Compute an adversarial ansatz given...

    Parameters:
    -----------
        adv_antz_data (np arr) : data used to generate the ansatz
        K             (np arr) : smearing matrix
        s_means       (np arr) : smear bin means
        bin_lb        (float)  : min lower bound for all true bins
        bin_ub        (float)  : max upper bound for all true bins
        dim_true      (int)    : dimension of the true space

    Returns:
    --------
        intensity_min_min (func) : the adversarial ansatz function
    """
    x_opt_min_min = constrained_ls_estimator_gmm_only(
        data=adv_antz_data, K=K, smear_means=s_means
    )

    # fit the cubic-spline to the above
    true_edges_fr = compute_even_space_bin_edges(
        bin_lb=bin_lb, bin_ub=bin_ub, num_bins=dim_true
    )

    intensity_min_min = fit_interpolator_intensity(
        true_edges=true_edges_fr,
        ls_est_vals=x_opt_min_min
    )

    return intensity_min_min


if __name__ == "__main__":

    # base directories
    BASE_DATA_DIR = './data'

    # read in parameter values
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

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
    print('Finding adversarial ansatz...')

    intensity_min_min = compute_adversarial_ansatz(
        adv_antz_data=ansatz_data[min_min_idx, :],
        K=K_fr,
        s_means=s_means_fr,
        bin_lb=-7,
        bin_ub=7,
        dim_true=40
    )

    true_edges_fr = compute_even_space_bin_edges(
        bin_lb=-7, bin_ub=7, num_bins=40
    )
    true_edges_rd = compute_even_space_bin_edges(
        bin_lb=-7, bin_ub=7, num_bins=80
    )

    # compute the ansatz matrix
    print('Computing full-rank adversarial ansatz matrix...')
    K_ansatz_min_min = compute_K_arbitrary(
        intensity_func=intensity_min_min,
        s_edges=true_edges_fr,  # same as true edges in the full-rank setup
        t_edges=true_edges_fr,
        sigma_smear=parameters['smear_strength']
    )
    
    print('Computing rank-deficient adversarial ansatz matrix...')
    intensity_f_rd = partial(intensity_min_min_const, intensity_f=intensity_min_min, const=10)
    K_ansatz_min_min_rd = compute_K_arbitrary(
        intensity_func=intensity_f_rd,
        s_edges=true_edges_fr,  # same as true edges in the full-rank setup
        t_edges=true_edges_rd,
        sigma_smear=parameters['smear_strength']
    )

    # compute the true bin ansatz means
    print('Computing full-rank adversarial ansatz true and smear means...')
    ansatz_true_means, ansatz_smear_means = compute_adversarial_bin_means(
        true_edges=true_edges_fr,
        intensity_min_min=intensity_min_min,
        K_ansatz=K_ansatz_min_min
    )

    print('Computing rank-deficient adversarial ansatz true and smear means...')
    ansatz_true_means_rd, ansatz_smear_means_rd = compute_adversarial_bin_means(
        true_edges=true_edges_rd,
        intensity_min_min=intensity_min_min,
        K_ansatz=K_ansatz_min_min_rd
    )

    # save the above
    np.savez(
        file=BASE_DATA_DIR + '/brute_force_ansatz/adversarial_ansatz_matrices_and_bin_means.npz',
        min_min_idx=min_min_idx,
        K_ansatz_min_min=K_ansatz_min_min,
        K_ansatz_min_min_rd=K_ansatz_min_min_rd,
        ansatz_unfold_means=ansatz_true_means,
        ansatz_smear_means=ansatz_smear_means,
        ansatz_true_means_rd=ansatz_true_means_rd,
        ansatz_smear_means_rd=ansatz_smear_means_rd
    )
