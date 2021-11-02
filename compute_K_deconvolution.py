"""
Compute smearing matrices for wide-bin deconvolution simulations.

This script computes smearing matrices for the following setups (smear x true)
1. wide           (40 x 10)
2. Full Rank      (40 x 40)
3. Rank Deficient (40 x 80)

Usage:
    Under "switches of matrices to compute", toggle the desired matrix
    configurations you want to compute. These take on the order of several
    minutes, which is why the switches exist.

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 02 Nov 2021
===============================================================================
"""
import json
import numpy as np
from utils import (
    compute_even_space_bin_edges, compute_K_gmm, intensity_f
)
from time import time

def compute_true_and_mc_K(
    true_params, mc_params, true_edges, smear_edges, smear_strength
):
    """
    Computes both the true and monte carlo smearing matrices. This function
    expects the parameters of the GMM.

    Parameters:
    -----------
        true_params    (dict)   : parameters of the true GMM
        mc_params      (dict)   : parameters of the monte carlo GMM
        true_edges     (np arr) : edges of true bins
        smear_edges    (np arr) : edges of smear bins
        smear_strength (float)  : strength of the smearing

    Returns:
    --------
        K_true (np arr)
        K_mc   (np arr)
    """
    # determine the problem dimensions
    dim_smear = smear_edges.shape[0] - 1
    dim_true = true_edges.shape[0] - 1

    # pull out parameters
    pi_true = true_params['pi']
    mu_true = true_params['mu']
    sigma_true = true_params['sigma']
    T_true = true_params['T']

    pi_mc = mc_params['pi']
    mu_mc = mc_params['mu']
    sigma_mc = mc_params['sigma']
    T_mc = mc_params['T']

    # true smearing matrix
    K_true = compute_K_gmm(
        intensity_func=intensity_f,
        dim_smear=dim_smear, dim_true=dim_true,
        s_edges=smear_edges,
        t_edges=true_edges,
        sigma_smear=smear_strength,
        pi=pi_true, mu=mu_true, sigma=sigma_true, T=T_true
    )

    # monte carlo smearing matrix
    K_mc = compute_K_gmm(
        intensity_func=intensity_f,
        dim_smear=dim_smear, dim_true=dim_true,
        s_edges=smear_edges,
        t_edges=true_edges,
        sigma_smear=smear_strength,
        pi=pi_mc, mu=mu_mc, sigma=sigma_mc, T=T_mc
    )

    return K_true, K_mc


if __name__ == "__main__":

    # save locations
    MATRIX_BASE_LOC = './smearing_matrices'

    # switches of matrices to compute
    COMPUTE_WIDE = False
    COMPUTE_FR = False
    COMPUTE_RD = True

    # read in parameter values
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    bin_lb = parameters['wide_bin_deconvolution_bins']['bin_lb']
    bin_ub = parameters['wide_bin_deconvolution_bins']['bin_ub']

    # compute bin edges
    TRUE_BINS_WIDE = 10
    TRUE_BINS_FR = 40    # FR == "Full Rank"
    TRUE_BINS_RD = 80    # RD == "Rank Deficient"
    SMEAR_BINS = 40
    true_edges_wide = true_edges = compute_even_space_bin_edges(
        bin_lb=bin_lb, bin_ub=bin_ub, num_bins=TRUE_BINS_WIDE
    )
    true_edges_fr = compute_even_space_bin_edges(
        bin_lb=bin_lb, bin_ub=bin_ub, num_bins=TRUE_BINS_FR
    )
    true_edges_rd = compute_even_space_bin_edges(
        bin_lb=bin_lb, bin_ub=bin_ub, num_bins=TRUE_BINS_RD
    )
    smear_edges = compute_even_space_bin_edges(
        bin_lb=bin_lb, bin_ub=bin_ub, num_bins=SMEAR_BINS
    )

    # compute smearing matrices
    print("Computing smearing matrices...")

    # Wide
    if COMPUTE_WIDE:
        START = time()
        K_wide, K_wide_mc = compute_true_and_mc_K(
            true_params=parameters['gmm_truth'],
            mc_params=parameters['gmm_ansatz'],
            true_edges=true_edges_wide,
            smear_edges=smear_edges,
            smear_strength=parameters['smear_strength']
        )
        np.savez(
            file=MATRIX_BASE_LOC + '/K_wide_mats.npz',
            K_wide=K_wide,
            K_wide_mc=K_wide_mc
        )
        print('- Wide Setup [Done] -> %.2f seconds' % (time() - START))
    else:
        print('- Wide Setup [Skip]')

    # Full Rank
    if COMPUTE_FR:
        START = time()
        K_fr, K_fr_mc = compute_true_and_mc_K(
            true_params=parameters['gmm_truth'],
            mc_params=parameters['gmm_ansatz'],
            true_edges=true_edges_fr,
            smear_edges=smear_edges,
            smear_strength=parameters['smear_strength']
        )
        np.savez(
            file=MATRIX_BASE_LOC + '/K_full_rank_mats.npz',
            K_fr=K_fr,
            K_fr_mc=K_fr_mc
        )
        print('- Full Rank [Done] -> %.2f seconds' % (time() - START))
    else:
        print('- Full Rank [Skip]')

    # Rank Deficient
    if COMPUTE_RD:
        START = time()
        K_rd, K_rd_mc = compute_true_and_mc_K(
            true_params=parameters['gmm_truth'],
            mc_params=parameters['gmm_ansatz'],
            true_edges=true_edges_rd,
            smear_edges=smear_edges,
            smear_strength=parameters['smear_strength']
        )
        np.savez(
            file=MATRIX_BASE_LOC + '/K_rank_deficient_mats.npz',
            K_rd=K_rd,
            K_rf_mc=K_rd_mc
        )
        print('- Rank Deficient [Done] -> %.2f seconds' % (time() - START))
    else:
        print('- Rank Deficient [Skip]')
    
    print('---- Done ----')
