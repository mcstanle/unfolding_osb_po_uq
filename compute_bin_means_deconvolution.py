"""
Compute bin means for the deconvolution example for both the true intensity
functions and ansatz intensity functions.

True Bin Dimensions
1. Wide           (10)
2. Full Rank      (40)
3. Rank Deficient (80)

Intensity Functions
1. Truth
2. GMM Ansatz
3. Adversarial Ansatz

NOTE: there is no toggle in this script like the compute_K_deconvolution.py
file since computing these bin means takes little time (order of a few seconds)

Author        : Michael Stanley
Created       : 02 Nov 2021
Last Modified : 02 Nov 2021
===============================================================================
"""
from functools import partial
import json
import numpy as np
from utils import compute_GMM_bin_means, intensity_f


def compute_true_ansatz_GMM_means(
    true_edges,
    pi, mu, sigma, T, K,
    pi_ans, mu_ans, sigma_ans, T_ans, K_ans
):
    """
    Computes the true and smear bin means for both the true intensity and the
    ansatz intensity.

    Parameters:
    -----------
        true_edges (np arr) : edges of bins for true histogram
        pi         (np arr) : true mixing probabilties for each gaussian
        mu         (np arr) : true mean for each gaussian component
        sigma      (np arr) : true standard deviation for each component
        T          (np arr) : true Mean of poisson process
        K          (np arr) : true smearing matrix
        pi_ans     (np arr) : ansatz mixing probabilties for each gaussian
        mu_ans     (np arr) : ansatz mean for each gaussian component
        sigma_ans  (np arr) : ansatz standard deviation for each component
        T_ans      (np arr) : ansatz Mean of poisson process
        K_ans      (np arr) : ansatz smearing matrix

    Returns:
    --------
        true_means         (np arr)
        smear_means        (np arr)
        true_means_ansatz  (np arr)
        smear_means_ansatz (np arr)
    """
    # compute bin means for truth
    true_means, smear_means = compute_GMM_bin_means(
        true_edges=true_edges,
        intensity_func=intensity_f,
        pi=pi,
        mu=mu,
        sigma=sigma,
        T=T,
        K=K
    )
    
    # compute bin means for ansatz
    true_means_ansatz, smear_means_ansatz = compute_GMM_bin_means(
        true_edges=true_edges,
        intensity_func=intensity_f,
        pi=pi_ans,
        mu=mu_ans,
        sigma=sigma_ans,
        T=T_ans,
        K=K_ans
    )

    return true_means, smear_means, true_means_ansatz, smear_means_ansatz

if __name__ == "__main__":

    # save locations
    BIN_MEAN_BASE_LOC = './bin_means'

    # matrix locations
    MATRIX_BASE_LOC = './smearing_matrices'    

    # read in parameter values
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    bin_lb = parameters['wide_bin_deconvolution_bins']['bin_lb']
    bin_ub = parameters['wide_bin_deconvolution_bins']['bin_ub']

    # read in the smearing matrices
    K_wide_objs = np.load(file=MATRIX_BASE_LOC + '/K_wide_mats.npz')
    K_fr_objs = np.load(file=MATRIX_BASE_LOC + '/K_full_rank_mats.npz')
    K_rd_objs = np.load(file=MATRIX_BASE_LOC + '/K_rank_deficient_mats.npz')

    K_wide = K_wide_objs['K_wide']
    K_wide_mc = K_wide_objs['K_wide_mc']
    K_fr = K_fr_objs['K_fr']
    K_fr_mc = K_fr_objs['K_fr_mc']
    K_rd = K_rd_objs['K_rd']
    K_rd_mc = K_rd_objs['K_rd_mc']

    # compute the true bin edges
    TRUE_BINS_WIDE = 10
    TRUE_BINS_FR = 40    # FR == "Full Rank"
    TRUE_BINS_RD = 80    # RD == "Rank Deficient"

    true_edges_wide = np.linspace(bin_lb, bin_ub, TRUE_BINS_WIDE + 1)
    true_edges_fr = np.linspace(bin_lb, bin_ub, TRUE_BINS_FR + 1)
    true_edges_rd = np.linspace(bin_lb, bin_ub, TRUE_BINS_RD + 1)

    # compute the bin means
    TRUE_PARAMS = parameters['gmm_truth']
    ANSATZ_PARAMS = parameters['gmm_ansatz']

    compute_GMM_means_par = partial(
        compute_true_ansatz_GMM_means,
        pi=TRUE_PARAMS['pi'],
        mu=TRUE_PARAMS['mu'],
        sigma=TRUE_PARAMS['sigma'],
        T=TRUE_PARAMS['T'],
        pi_ans=ANSATZ_PARAMS['pi'],
        mu_ans=ANSATZ_PARAMS['mu'],
        sigma_ans=ANSATZ_PARAMS['sigma'],
        T_ans=ANSATZ_PARAMS['T'],
    )

    # Wide
    print("Computing bin means...")

    t_means_w, s_means_w, t_means_ansatz_w, s_means_ansatz_w = compute_GMM_means_par(
        true_edges=true_edges_wide,
        K=K_wide,
        K_ans=K_wide_mc
    )
    print('- Wide Setup [Done]')

    # Full Rank
    t_means_fr, s_means_fr, t_means_ansatz_fr, s_means_ansatz_fr = compute_GMM_means_par(
        true_edges=true_edges_fr,
        K=K_fr,
        K_ans=K_fr_mc
    )
    print('- Full Rank Setup [Done]')

    # Rank Deficient
    t_means_rd, s_means_rd, t_means_ansatz_rd, s_means_ansatz_rd = compute_GMM_means_par(
        true_edges=true_edges_rd,
        K=K_rd,
        K_ans=K_rd_mc
    )
    print('- Rank Deficient Setup [Done]')

    # save the above
    np.savez(
        file=BIN_MEAN_BASE_LOC + '/gmm_wide.npz',
        t_means_w=t_means_w,
        s_means_w=s_means_w,
        t_means_ansatz_w=t_means_ansatz_w,
        s_means_ansatz_w=s_means_ansatz_w
    )
    np.savez(
        file=BIN_MEAN_BASE_LOC + '/gmm_fr.npz',
        t_means_fr=t_means_fr,
        s_means_fr=s_means_fr,
        t_means_ansatz_fr=t_means_ansatz_fr,
        s_means_ansatz_fr=s_means_ansatz_fr
    )
    np.savez(
        file=BIN_MEAN_BASE_LOC + '/gmm_rd.npz',
        t_means_rd=t_means_rd,
        s_means_rd=s_means_rd,
        t_means_ansatz_rd=t_means_ansatz_rd,
        s_means_ansatz_rd=s_means_ansatz_rd
    )