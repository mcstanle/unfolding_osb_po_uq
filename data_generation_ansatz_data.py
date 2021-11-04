"""
Generate data draws from the true poisson process as candidate data for fitting
the adversarial ansatzs.

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 04 Nov 2021
===============================================================================
"""
import json
import numpy as np
from scipy import stats

if __name__ == "__main__":

    # simulation operational parameters
    NUM_SIMS = 1000

    # read in true bin means
    bin_means_obj = np.load(file='./bin_means/gmm_fr.npz')
    s_means_fr = bin_means_obj['s_means_fr']

    # generate the data
    dim_smear = s_means_fr.shape[0]
    data_true = np.zeros(shape=(NUM_SIMS, dim_smear))
        
    for j in range(dim_smear):

        # generate data
        data_true[:, j] = stats.poisson(s_means_fr[j]).rvs(NUM_SIMS)

    # save the above
    np.save(
        file='./data/brute_force_ansatz/ansatz_data_gmm_NEW.npy',
        arr=data_true
    )