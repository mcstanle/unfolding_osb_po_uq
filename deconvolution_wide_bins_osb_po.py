"""
Executes the simulation studies of the OSB and PO intervals under a variety
of settings.

There are two considered settings:
1. GMM Ansatz
2. Adversarial Ansatz

Author        : Michael Stanley
Created       : 06 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
from interval_estimators import osb_interval, po_interval
import numpy as np
from tqdm import tqdm
from utils import compute_coverage


def run_osb_interval_exp(data, H, smear_means, K, A, alpha):
    """
    Compute an ensemble of bin-wise OSB intervals.

    Parameters:
    -----------
        data        (np arr) : ensemble of data draws
        H           (np arr) : collection of linear functionals
        smear_means (np arr) : smear bin means to compute Cholesky decomp
        K           (np arr) : smearing matrix
        A           (np arr) : constraint matrix
        alpha       (np arr) : type 1 error threshold -- (1 - confidence level)

    Returns:
    --------
        intervals (np arr) : the collection of optimized intervals
    """
    num_sims = data.shape[0]
    num_funcs = H.shape[0]

    intervals = np.zeros(shape=(num_sims, num_funcs, 2))

    # find the change in basis
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)

    # transform the matrix
    K_mc_tilde = L_data_inv @ K

    for i in tqdm(range(num_sims)):
        
        # transform the data
        data_i = L_data_inv @ data[i, :]
        
        for j in range(num_funcs):

            # fit the intervals
            interval_ij = osb_interval(
                y=data_i, K=K_mc_tilde, h=H[j, :], A=A, alpha=alpha
            )
            
            # save the intervals for later
            intervals[i, j, :] = interval_ij

    return intervals


def run_po_interval_exp(prior, data, H, smear_means, K, A, alpha):
    """
    Compute an ensemble of bin-wise PO intervals.

    NOTE: Since the PO intervals are essentially optimized without the data,
    this function could be made substantially faster by performing the
    optimization and then iterating over the data. One would use the
    "return_int" argument in the po_interval function.

    Parameters:
    -----------
        prior       (np arr) : prior used to optimize the intervals
        data        (np arr) : ensemble of data draws
        H           (np arr) : collection of linear functionals
        smear_means (np arr) : smear bin means to compute Cholesky decomp
        K           (np arr) : smearing matrix
        A           (np arr) : constraint matrix
        alpha       (np arr) : type 1 error threshold -- (1 - confidence level)

    Returns:
    --------
        intervals (np arr) : the collection of optimized intervals
    """
    num_sims = data.shape[0]
    num_funcs = H.shape[0]

    intervals = np.zeros(shape=(num_sims, num_funcs, 2))

    # find the change in basis
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)

    # transform the matrix
    K_mc_tilde = L_data_inv @ K

    for i in tqdm(range(num_sims)):
        
        # transform the data
        data_i = L_data_inv @ data[i, :]
        
        for j in range(num_funcs):

            # fit the interval
            interval_ij = po_interval(
                y=data_i, prior_mean=prior_40, K=K_mc_tilde,
                h=H[j, :], A=A, alpha=alpha
            )
            
            # save the intervals for later
            intervals[i, j, :] = interval_ij

    return intervals


if __name__ == "__main__":

    # operational switches
    GMM_ANSATZ = True
    ADVERSARIAL_ANSATZ = False
    READ_INTERVALS = True  # use this flag to exactly reproduce the paper results

    # interval parameters
    ALPHA = 0.05

    # read in the true aggregated bin means
    t_means_w = np.load(file='./bin_means/gmm_wide.npz')['t_means_w']

    if GMM_ANSATZ:

        # fit the ensemble of intervals
        if READ_INTERVALS:  # this is the original ensemble that created results in paper
            intervals_full_rank_gmm_ans_files = np.load(
                file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz_ORIGINAL.npz'
            )
            intervals_osb_fr = intervals_full_rank_gmm_ans_files['intervals_osb_fr']
            intervals_po_fr = intervals_full_rank_gmm_ans_files['intervals_po_fr']
        else:
            
            # import the data
            data = np.load(file='./data/wide_bin_deconvolution/simulation_data_ORIGINAL.npy')

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

            # OSB intervals
            intervals_osb_fr = run_osb_interval_exp(
                data=data,
                H=H,
                smear_means=s_means_fr,
                K=K_fr_mc,
                A=-np.identity(t_means_fr.shape[0]),  # positivity constraint, only
                alpha=ALPHA
            )

            # PO intervals
            prior_40 = np.ones(40) * t_means_fr.mean()
            intervals_po_fr = run_po_interval_exp(
                prior=prior_40,
                data=data,
                H=H,
                smear_means=s_means_fr,
                K=K_fr_mc,
                A=-np.identity(t_means_fr.shape[0]),  # positivity constraint, only
                alpha=ALPHA
            )

            # save the above intervals
            np.savez(
                file=BASE_DIR + './data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz.npz',
                intervals_ls_fr=intervals_ls_agg,
                intervals_osb_fr=intervals_osb_fr,
                intervals_po_fr=intervals_po_fr,
            )

        # estimate the coverage
        coverage_osb_fr = compute_coverage(
            intervals=intervals_osb_fr,
            true_bin_means=t_means_w
        )
        coverage_po_fr = compute_coverage(
            intervals=intervals_po_fr,
            true_bin_means=t_means_w
        )

        # save the above
        np.savez(
            file='./data/wide_bin_deconvolution/coverage_osb_po_full_rank_misspec_gmm_ansatz',
            coverage_osb_fr=coverage_osb_fr,
            coverage_po_fr=coverage_po_fr
        )