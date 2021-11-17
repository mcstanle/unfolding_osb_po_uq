"""
Compute bin means for the steeply falling spectrum example for both the true
intensity functions and ansatz intensity functions.

True Bin Dimensions
1. Wide           (10)
2. Rank Def       (60)

Intensity Functions
1. Truth
2. Ansatz

Author        : Michael Stanley
Created       : 16 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
from functools import partial
import json
import numpy as np
from steeply_falling_spectra import (
    compute_intensities, compute_means, f_sf
)
from time import time


def compute_true_ansatz_sfs_means(
    true_edges, smear_edges, f_true, f_ansatz
):
    """
    Computes the true and smear bin means for both the true intensity and the
    ansatz intensity.

    Parameters:
    -----------
        true_edges  (np arr) : edges of bins for true histogram
        smear_edges (np arr) : edges of bins for smeared histogram
        f_true      (func)   : real function for true intensity
        f_ansatz    (func)   : real function for ansatz intensity

    Returns:
    --------
        true_means         (np arr)
        smear_means        (np arr)
        true_means_ansatz  (np arr)
        smear_means_ansatz (np arr)
    """
    # true intensity means
    true_means, smear_means = compute_means(
        f_intensity=f_true,
        true_grid=true_edges,
        smear_grid=smear_edges
    )

    # ansatz intensity means
    true_means_ans, smear_means_ans = compute_means(
        f_intensity=f_ansatz,
        true_grid=true_edges,
        smear_grid=smear_edges
    )

    return true_means, smear_means, true_means_ans, smear_means_ans


if __name__ == "__main__":

    # save locations
    BIN_MEAN_BASE_LOC = './bin_means'

    # create bin grids
    true_grid = np.linspace(start=400, stop=1000, num=60 + 1)
    true_grid_wide = np.linspace(start=400, stop=1000, num=11)
    smear_grid = np.linspace(start=400, stop=1000, num=30 + 1)

    # define the intensity functions
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    true_params = parameters['steep_f_spec_truth']
    mc_params = parameters['steep_f_spec_ansatz']

    f_true, f_ans = compute_intensities()

    # wide setup
    t_means_w, s_means_w, t_means_ans_w, s_means_ans_w = compute_true_ansatz_sfs_means(
        true_edges=true_grid_wide,
        smear_edges=smear_grid,
        f_true=f_true,
        f_ansatz=f_ans
    )

    # Rank-deficient setup
    t_means, s_means, t_means_ans, s_means_ans = compute_true_ansatz_sfs_means(
        true_edges=true_grid,
        smear_edges=smear_grid,
        f_true=f_true,
        f_ansatz=f_ans
    )

    # save the above
    np.savez(
        file=BIN_MEAN_BASE_LOC + '/sfs_wide.npz',
        t_means_w=t_means_w,
        s_means_w=s_means_w,
        t_means_ans_w=t_means_ans_w,
        s_means_ans_w=s_means_ans_w
    )
    np.savez(
        file=BIN_MEAN_BASE_LOC + '/sfs_rd.npz',
        t_means=t_means,
        s_means=s_means,
        t_means_ans=t_means_ans,
        s_means_ans=s_means_ans
    )
