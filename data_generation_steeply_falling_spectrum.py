"""
Script to create the data draws from the steeply falling particle spectrum in
section 5.

NOTE: To exactly reproduce the results from the paper, please use the data
./data/steeply_falling_spectrum/data_ORIGINAL.npy. Those data were
generated using the code below, so if the code below is used, the downstream
results should still be very close.

Author        : Michael Stanley
Created       : 17 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
import numpy as np
from steeply_falling_spectra import gen_poisson_data
from tqdm import tqdm

if __name__ == "__main__":

    # operational parameters
    NUM_DATA = 10000

    # read in the true bin means
    true_means = np.load(file='./bin_means/sfs_rd.npz')['t_means']

    # read in the true smearing matrix
    K = np.load(file='./smearing_matrices/K_rank_def_mats_sfs.npz')['K']

    # generate the data
    data = np.zeros(shape=(NUM_DATA, 30))

    for i in tqdm(range(NUM_DATA)):
        data[i, :] = gen_poisson_data(mu=K @ true_means)

    # save the above
    np.save(file='./data/steeply_falling_spectrum/data.npy', arr=data)