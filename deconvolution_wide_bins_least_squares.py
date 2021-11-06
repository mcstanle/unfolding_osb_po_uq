"""
Executes the simulation studies of the least-squares intervals under a variety
of settings.

Author        : Michael Stanley
Created       : 05 Nov 2021
Last Modified : 05 Nov 2021
===============================================================================
"""
from interval_estimators import (
    least_squares_interval
)
import numpy as np
from tqdm import tqdm
from utils import (
    compute_coverage
)

def run_ls_coverage_exp(
    num_sims, true_means, smear_means, K, data, H, alpha
):
    """
    Run the least-squares coverage experiment.

    Parameters:
    -----------
        num_sims    (int)    : number of simulations to estimate coverage
        true_means  (np arr) : true bin means
        smear_means (np arr) : smear bin means to use in the Gaussian approx.
        K           (np arr) : smearing matrix
        data        (np arr) : data used to estimate coverage
        H           (np arr) : matrix of functionals
        alpha       (np arr) : type 1 error threshold -- (1 - confidence level)

    Returns:
    --------
        intervals (np arr) : the ensemble of fitted intervals
        coverage  (np arr) : the estimated bin-wise coverage
    """
    num_funcs = H.shape[0]

    # find the least squares intervals
    intervals = np.zeros(shape=(num_sims, num_funcs, 2))

    # perform the data transformation
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)

    # transform the matrix
    K_tilde_i = L_data_inv @ K

    for i in tqdm(range(num_sims)):
        
        # transform the data
        data_i = L_data_inv @ data[i, :]
        
        for j in range(num_funcs):
            intervals[i, j, :] = least_squares_interval(
                K=K_tilde_i, h=H[j, :],
                y=data_i, alpha=ALPHA
            )

    # compute the coverage of the above intervals
    true_bin_means_agg = H @ true_means
    coverage = compute_coverage(
        intervals=intervals,
        true_bin_means=true_bin_means_agg
    )

    return intervals, coverage


if __name__ == "__main__":

    # operational parameters and switches
    NUM_SIMS = 1000
    ALPHA = 0.05
    RUN_WIDE_LS = False
    RUN_FR_FB_LS = False  # FR = "Full Rank", FB = "Fine Bin"
    RUN_FR_AGG_LS = True  # FR = "Full Rank", AGG = "Aggregated"

    # import the data
    data = np.load(file='./data/wide_bin_deconvolution/simulation_data_ORIGINAL.npy')

    if RUN_WIDE_LS:  # mades ensemble of intervals for figure 3

        # import the smearing matrix
        K_wide_mc = np.load(
            file='./smearing_matrices/K_wide_mats.npz'
        )['K_wide_mc']

        # import the bin means
        bin_means_obj = np.load(file='./bin_means/gmm_wide.npz')
        t_means_w = bin_means_obj['t_means_w']
        s_means_w = bin_means_obj['s_means_w']

        H_wb = np.identity(10)  # since we are unfolding directly to wide-bins
        ints_ls_wb, coverage_ls_wb = run_ls_coverage_exp(
            num_sims=NUM_SIMS,
            true_means=t_means_w,
            smear_means=s_means_w,
            K=K_wide_mc,
            data=data,
            H=H_wb,
            alpha=ALPHA
        )

        # save intervals and coverage
        np.savez(
            file='./data/wide_bin_deconvolution/ints_cov_wide_ls.npz',
            intervals=ints_ls_wb,
            coverage=coverage_ls_wb
        )

    if RUN_FR_FB_LS:  # mades ensemble of intervals for figure 4

        # import the smearing matrix
        K_fr_mc = np.load(
            file='./smearing_matrices/K_full_rank_mats.npz'
        )['K_fr_mc']

        # import the bin means
        bin_means_obj = np.load(file='./bin_means/gmm_fr.npz')
        t_means_fr = bin_means_obj['t_means_fr']
        s_means_fr = bin_means_obj['s_means_fr']

        H_fb = np.identity(40)  # since we are unfolding directly to fine-bins
        ints_ls_fb, coverage_ls_fb = run_ls_coverage_exp(
            num_sims=NUM_SIMS,
            true_means=t_means_fr,
            smear_means=s_means_fr,
            K=K_fr_mc,
            data=data,
            H=H_fb,
            alpha=ALPHA
        )

        # save intervals and coverage
        np.savez(
            file='./data/wide_bin_deconvolution/ints_cov_fine_ls.npz',
            intervals=ints_ls_fb,
            coverage=coverage_ls_fb
        )

    if RUN_FR_AGG_LS:  # mades ensemble of intervals for figure 5

        # import the smearing matrix
        K_fr_mc = np.load(
            file='./smearing_matrices/K_full_rank_mats.npz'
        )['K_fr_mc']

        # import the bin means
        bin_means_obj = np.load(file='./bin_means/gmm_fr.npz')
        t_means_fr = bin_means_obj['t_means_fr']
        s_means_fr = bin_means_obj['s_means_fr']

        # import the aggregating functionals
        H = np.load(file='./functionals/H_deconvolution.npy')

        ints_ls_agg, coverage_ls_agg = run_ls_coverage_exp(
            num_sims=NUM_SIMS,
            true_means=t_means_fr,
            smear_means=s_means_fr,
            K=K_fr_mc,
            data=data,
            H=H,
            alpha=ALPHA
        )

        # save intervals and coverage
        np.savez(
            file='./data/wide_bin_deconvolution/ints_cov_agg_ls.npz',
            intervals=ints_ls_agg,
            coverage=coverage_ls_agg
        )