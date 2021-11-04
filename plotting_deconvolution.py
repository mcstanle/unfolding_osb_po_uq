"""
Script containing all code to generate the plots for section 4.

This script is written so that each function can plot its respective figure in
isolation from any other figure. This way, one can simply toggle the figures
one wants to plot in the main section of the code.

Author        : Michael Stanley
Created       : 04 Nov 2021
Last Modified : 04 Nov 2021
===============================================================================
"""
from compute_K_deconvolution_adversarial_ansatz import compute_adversarial_ansatz
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import intensity_f

plt.style.use('seaborn-colorblind')

def plot_figure1(save_loc=None):
    """
    Illustration of the true intensity function used for simulations and the
    two ansatz functions used for computing the smearing matrix K.

    Parameters:
    -----------
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
    """
    # read in the GMM intensity function parameters
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    true_params = parameters['gmm_truth']
    gmm_ansatz_params = parameters['gmm_ansatz']

    # compute the adversarial ansatz
    K_fr = np.load(file='./smearing_matrices/K_full_rank_mats.npz')['K_fr']
    s_means_fr = np.load(file='./bin_means/gmm_fr.npz')['s_means_fr']
    ansatz_data = np.load('./data/brute_force_ansatz/ansatz_data_gmm.npy')
    min_min_idx = np.load(
        './data/brute_force_ansatz/adversarial_ansatz_matrices_and_bin_means.npz'
    )['min_min_idx']
    intensity_min_min = compute_adversarial_ansatz(
        adv_antz_data=ansatz_data[min_min_idx, :],
        K=K_fr,
        s_means=s_means_fr,
        bin_lb=-7,
        bin_ub=7,
        dim_true=40
    )

    plt.figure(figsize=(10, 5))

    x_vals = np.linspace(-7, 7, num=200)

    # true intensity
    plt.plot(
        x_vals,
        [intensity_f(
            x=x_i,
            pi=true_params['pi'], mu=true_params['mu'],
            sigma=true_params['sigma'], T=true_params['T']
            ) for x_i in x_vals
        ],
        label='True Intensity'
    )

    # Misspecified GMM
    plt.plot(
        x_vals,
        [intensity_f(
            x=x_i,
            pi=gmm_ansatz_params['pi'], mu=gmm_ansatz_params['mu'],
            sigma=gmm_ansatz_params['sigma'], T=gmm_ansatz_params['T']
            ) for x_i in x_vals
        ],
        label='Misspecified GMM'
    )

    # Adversarial Ansatz
    plt.plot(
        x_vals, [intensity_min_min(x_i) for x_i in x_vals],
        label='Adversarial'
    )

    # axis labels
    plt.ylabel('Intensity')
    plt.xlabel('Physical Observable')

    plt.legend()
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()
