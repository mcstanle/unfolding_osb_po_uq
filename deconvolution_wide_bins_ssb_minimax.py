"""
Executes the simulation studies involving the SSB and minimax intervals on
the wide-bin deconvolution problem.

Author        : Michael Stanley
Created       : 06 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
from interval_estimators import minimax_interval_radius_bounds, ssb_interval
import numpy as np
from tqdm import tqdm
from utils import compute_coverage


def compute_minimax_bin_bounds(H, K, smear_means, alpha):
    """
    Computes the minimax upper and lower bounds for each bin.

    Parameters:
    -----------
        H           (np arr) : bin aggregation functionals
        K           (np arr) : smearing matrix
        smear_means (np arr) : smear bin means
        alpha       (float)  : 1 - confidence level

    Returns:
    --------
        minimax_interval_widths (np arr) : bin-wise width bounds
    """
    num_funcs = H.shape[0]
    minimax_interval_widths = np.zeros(shape=(10, 2))

    # compute the cholesky transform
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)

    # transform the matrix
    K_tilde = L_data_inv @ K

    for j in tqdm(range(num_funcs)):
        lb_len, ub_len, z_lb, z_ub = minimax_interval_radius_bounds(
            h=H[j, :], 
            K=K_tilde,
            alpha=alpha
        )
            
        minimax_interval_widths[j] = [lb_len, ub_len]

    return minimax_interval_widths


def run_ssb_interval_exp(data, H, K, A, true_means, smear_means, alpha):
    """
    Run SSB interval ensemble experiment. Finds ensemble of intervals and
    estimates bin-wise coverage.

    Parameters:
    -----------
        data        (np arr) : ensemble of data
        H           (np arr) : bin aggregation functionals
        K           (np arr) : smearing matrix
        A           (np arr) : constraint matrix
        true_means  (np arr) : true aggreged bin means
        smear_means (np arr) : smear bin means
        alpha       (float)  : 1 - confidence level

    Returns:
    --------
        intervals (np arr) : ensemble of computed intervals
        coverage  (np arr) : estimated bin-wise coverage
    """
    num_sims = data.shape[0]
    num_funcs = H.shape[0]

    intervals = np.zeros(shape=(num_sims, num_funcs, 2))
    coverage = np.zeros(shape=(num_sims, num_funcs))

    # find the change in basis
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)

    # transform the matrix
    K_tilde = L_data_inv @ K

    for i in tqdm(range(num_sims)):

        # transform the data
        data_i = L_data_inv @ data[i, :]
        
        for j in range(num_funcs):

            # fit the interval
            intervals[i, j, :] = ssb_interval(
                y=data_i,
                K=K_tilde,
                h=H[j, :],
                A=A,
                alpha=alpha
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
    ALPHA = 0.05
    ENS_SIZE = 1000  # number of ensemble elements to use
    READ_INTERVALS = False

    if READ_INTERVALS:
        pass
    
    else:

        # import the data
        data = np.load(file='./data/wide_bin_deconvolution/simulation_data_ORIGINAL.npy')[:ENS_SIZE, :]

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

        # compute minimax bounds
        minimax_interval_widths = compute_minimax_bin_bounds(
            H=H,
            K=K_fr_mc,
            smear_means=s_means_fr,
            alpha=ALPHA
        )

        # compute SSB intervals
        ssb_intervals, ssb_coverage = run_ssb_interval_exp(
            data=data,
            H=H,
            K=K_fr_mc,
            A=-np.identity(t_means_fr.shape[0]),
            true_means=t_means_fr,
            smear_means=s_means_fr,
            alpha=ALPHA
        )

        # save the above results
        np.savez(
            file='./data/wide_bin_deconvolution/intervals_ssb_minimax_full_rank_misspec_gmm_ansatz.npz',
            minimax_interval_widths=minimax_interval_widths,
            ssb_intervals=ssb_intervals,
            ssb_coverage=ssb_coverage
        )
