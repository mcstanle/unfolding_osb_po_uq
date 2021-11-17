"""
Compute smearing matrices for steeply falling spectrum simulations.

Usage:
    Under "switches of matrices to compute", toggle the desired matrix
    configurations you want to compute. These take on the order of several
    minutes, which is why the switches exist.

This script computes smearing matrices for the following setups (smear x true)
1. Wide           (30 x 10)
2. Rank Deficient (30 x 30)

NOTE: matrices are saved with the suffix, 'sfs' for "steeply falling spectrum"

Author        : Michael Stanley
Created       : 02 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
from functools import partial
import json
import numpy as np
from steeply_falling_spectra import (
    compute_K, f_sf
)
from time import time

def compute_true_and_mc_K(true_params, mc_params, true_edges, smear_edges):
    """
    Computes both the true and monte carlo smearing matrices for the steeply
    falling particle spectrum example.

    Parameters:
    -----------
        true_params    (dict)   : true parameters
        mc_params      (dict)   : ansatz parameters
        true_edges     (np arr) : edges of true bins
        smear_edges    (np arr) : edges of smear bins

    Returns:
    --------
        K_true (np arr)
        K_ans  (np arr)
    """
    # create each intensity function
    f_true = partial(
        f_sf, 
        L=true_params['L'],
        N_0=true_params['N_0'],
        alpha=true_params['ALPHA'],
        sqrt_s=true_params['SQRT_S'],
        beta=true_params['BETA'],
        gamma=true_params['GAMMA']
    )
    f_ans = partial(
        f_sf, 
        L=mc_params['L'],
        N_0=mc_params['N_0'],
        alpha=mc_params['ALPHA'],
        sqrt_s=mc_params['SQRT_S'],
        beta=mc_params['BETA'],
        gamma=mc_params['GAMMA']
    )

    K_true = compute_K(
        f_intensity=f_true, s_edges=smear_edges, t_edges=true_edges
    )
    K_ans = compute_K(
        f_intensity=f_ans, s_edges=smear_edges, t_edges=true_edges
    )

    return K_true, K_ans


if __name__ == "__main__":

    # save locations
    MATRIX_BASE_LOC = './smearing_matrices'

    # switches of matrices to compute
    COMPUTE_WIDE = True
    COMPUTE_FR = True

    # create bin grids
    true_grid = np.linspace(start=400, stop=1000, num=60 + 1)
    true_grid_wide = np.linspace(start=400, stop=1000, num=11)
    smear_grid = np.linspace(start=400, stop=1000, num=30 + 1)

    # read in the parameter values
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    true_params = parameters['steep_f_spec_truth']
    mc_params = parameters['steep_f_spec_ansatz']  

    # compute matrices
    if COMPUTE_WIDE:
        START = time()
        K_wide, K_wide_mc = compute_true_and_mc_K(
            true_params=true_params,
            mc_params=mc_params,
            true_edges=true_grid_wide,
            smear_edges=smear_grid
        )
        np.savez(
            file=MATRIX_BASE_LOC + '/K_wide_mats_sfs.npz',
            K_wide=K_wide,
            K_wide_mc=K_wide_mc
        )
        print('- Wide Setup [Done] -> %.2f seconds' % (time() - START))
    else:
        print('- Wide Setup [Skip]')
    
    if COMPUTE_FR:
        START = time()
        K, K_mc = compute_true_and_mc_K(
            true_params=true_params,
            mc_params=mc_params,
            true_edges=true_grid,
            smear_edges=smear_grid
        )
        np.savez(
            file=MATRIX_BASE_LOC + '/K_rank_def_mats_sfs.npz',
            K=K,
            K_mc=K_mc
        )
        print('- Rank Deficient [Done] -> %.2f seconds' % (time() - START))
    else:
        print('- Rank Deficient [Skip]')
