"""
Executes the simulation studies for least squares intervals in the wide-bin
setup for the steeply falling spectrum experiment.

Author        : Michael Stanley
Created       : 17 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
from compute_K_steeply_falling_spectrum import (
    compute_true_and_mc_K
)
import json
from interval_estimators import least_squares_interval
import numpy as np
from tqdm import tqdm
from utils import int_covers_truth

if __name__ == "__main__":

    # read in the variable width true bin end points
    close_func_endpoints = np.load('./functionals/sfs_close_func_endpoints.npy')

    # define the smear bins endpoints
    smear_grid = np.linspace(start=400, stop=1000, num=30 + 1)

    # read in the parameter values
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    true_params = parameters['steep_f_spec_truth']
    mc_params = parameters['steep_f_spec_ansatz']  

    # compute the true and ansatz matrices
    K, K_mc = compute_true_and_mc_K(
        true_params=true_params,
        mc_params=mc_params,
        true_edges=close_func_endpoints,
        smear_edges=smear_grid
    )

    # read in the data
    data = np.load('./data/steeply_falling_spectrum/data_ORIGINAL.npy')

    # read in the variable bin-width functional
    H = np.load(file='./functionals/H_steeply_falling_spectrum.npy')

    # make functionals that just pick out one bin at a time
    H_wb = np.identity(10)

    # read in the true/smear bin means
    true_means = np.load(file='./bin_means/sfs_rd.npz')['t_means']
    smear_means = np.load(file='./bin_means/sfs_rd.npz')['s_means']

    # compute the true variable bin functional values
    true_func_vals = H @ true_means

    # perform cholesky transform     
    Sigma = np.diag(smear_means)
    L_chol = np.linalg.cholesky(a=Sigma)
    L_chol_inv = np.linalg.inv(L_chol)

    # transform the matrix
    K_mc_tilde = L_chol_inv @ K_mc

    # find the least squares intervals
    NUM_SIMS = 1000
    intervals_ls_wb = np.zeros(shape=(NUM_SIMS, 10, 2))
    coverage_ls_wb = np.zeros(shape=(NUM_SIMS, 10))

    for i in tqdm(range(NUM_SIMS)):
        
        # transform the data
        data_i = L_chol_inv @ data[i, :]

        for j in range(10):

            # compute interval
            intervals_ls_wb[i, j, :] = least_squares_interval(
                K=K_mc_tilde, h=H_wb[j, :], y=data_i, alpha=0.05
            )

            # determine if covers
            coverage_ls_wb[i, j] = int_covers_truth(true_func_vals[j], intervals_ls_wb[i, j, :])

    print('Computed Coverages: %s' % str(coverage_ls_wb.mean(axis=0)))

    # save the above
    np.savez(
        file='./data/steeply_falling_spectrum/ints_cov_wide_ls.npz',
        intervals_ls_wb=intervals_ls_wb,
        coverage_ls_wb=coverage_ls_wb
    )