"""
Script generating the results in section 4.6.2.

NOTE: to create the differently sized matrices and bin mean vectors, simply
modify the dimensions in compute_K_deconvolution.py and
compute_bin_means_deconvolution.py. 

Author        : Michael Stanley
Created       : 06 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
import numpy as np

if __name__ == "__main__":

    # operational parameters and switches
    ENS_SIZE = 1000
    ALPHA = 0.05

    load_objects = np.load(
        file='.v/data/wide_bin_deconvolution/true_and_mc_matrices_and_means_40_80_160_320_unfold_bins.npz'
    )

    K_40 = load_objects['K_40']
    K_80 = load_objects['K_80']
    K_160 = load_objects['K_160']
    K_320 = load_objects['K_320']
    K_40_mc = load_objects['K_40_mc']
    K_80_mc = load_objects['K_80_mc']
    K_160_mc = load_objects['K_160_mc']
    K_320_mc = load_objects['K_320_mc']
    true_means_40 = load_objects['true_means_40']
    true_means_80 = load_objects['true_means_80']
    true_means_160 = load_objects['true_means_160']
    true_means_320 = load_objects['true_means_320']
    smear_means_40 = load_objects['smear_means_40']
    smear_means_80 = load_objects['smear_means_80']
    smear_means_160 = load_objects['smear_means_160']
    smear_means_320 = load_objects['smear_means_320']
    true_means_mc_40 = load_objects['true_means_mc_40']
    true_means_mc_80 = load_objects['true_means_mc_80']
    true_means_mc_160 = load_objects['true_means_mc_160']
    true_means_mc_320 = load_objects['true_means_mc_320']
    smear_means_mc_40 = load_objects['smear_means_mc_40']
    smear_means_mc_80 = load_objects['smear_means_mc_80']
    smear_means_mc_160 = load_objects['smear_means_mc_160']
    smear_means_mc_320 = load_objects['smear_means_mc_320']


    # load data
    data = np.load(file='./data/wide_bin_deconvolution/simulation_data_ORIGINAL.npy')[:ENS_SIZE, :]

    # import the aggregating functionals
    H_40 = np.load(file='./functionals/H_deconvolution.npy')
    H_80 = np.load(file='./functionals/H_80_deconvolution.npy')
    H_160 = np.load(file='./functionals/H_160_deconvolution.npy')
    H_320 = np.load(file='./functionals/H_320_deconvolution.npy')

    # compute the true bin counts
    true_agg_40 = H_40 @ true_means_40