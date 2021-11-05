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
    int_covers_truth
)

def run_ls_coverage_exp(num_sims, smear_means, K, data, H, alpha):
    """
    Run the least-squares coverage experiment.

    Parameters:
    -----------
        num_sims    (int)    : number of simulations to estimate coverage
        smear_means (np arr) : smear bin means to use in the Gaussian approx.
        K           (np arr) : smearing matrix
        data        (np arr) : data used to estimate coverage
        H           (np arr) : matrix of functionals
        alpha       (np arr) : type 1 error threshold -- (1 - confidence level)

    Returns:
    --------
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

if __name__ == "__main__":

