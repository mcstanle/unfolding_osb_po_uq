"""
Script containing all code to generate the plots for section 4.

This script is written so that each function can plot its respective figure in
isolation from any other figure. This way, one can simply toggle the figures
one wants to plot in the main section of the code.

The convention is that the code to plot figure k can be found in function
called "plot_figurek". 

Author        : Michael Stanley
Created       : 04 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
from compute_K_deconvolution_adversarial_ansatz import compute_adversarial_ansatz
import json
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from utils import intensity_f, compute_mean_std_width

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
        None -- makes matplotlib plot
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


def plot_figure2(
    true_edges_wide=np.linspace(-7, 7, 11),
    smear_edges=np.linspace(-7, 7, 41),
    save_loc=None
):
    """
    True and smeared bin expected counts:

    Parameters:
    -----------
        true_edges_wide   (np arr) : edges of the wide true bins
        smear_edges       (np arr) : edges of the smeared bins
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    WIDTH_SMEAR = smear_edges[1] - smear_edges[0]
    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # read in the smear bin means data
    smear_bin_means_obj = np.load(file='./bin_means/gmm_fr.npz')
    s_means_gmm_ansatz_fr = smear_bin_means_obj['s_means_ansatz_fr']
    s_means_fr = smear_bin_means_obj['s_means_fr']

    # read in the true wide bin data
    true_bin_means_obj = np.load(file='./bin_means/gmm_wide.npz')
    true_means_w_ansatz = true_bin_means_obj['t_means_ansatz_w']
    true_means_w = true_bin_means_obj['t_means_w']

    # read in the ansatz data
    ansatz_data = np.load('./data/brute_force_ansatz/ansatz_data_gmm.npy')
    ansatz_obj = np.load(
        './data/brute_force_ansatz/adversarial_ansatz_matrices_and_bin_means.npz'
    )
    s_mean_adv_ansatz_fr = ansatz_obj['ansatz_smear_means']
    t_mean_adv_ansatz_fr = ansatz_obj['ansatz_true_means']
    min_min_idx = ansatz_obj['min_min_idx']

    # read in the aggregation functionals
    H = np.load(file='./functionals/H_deconvolution.npy')

    # smeared
    ax[0].bar(
        x=smear_edges[:-1],
        height=s_means_gmm_ansatz_fr,
        width=WIDTH_SMEAR,
        align='edge', fill=False, edgecolor='blue', label='Misspecified GMM'
    )
    ax[0].bar(
        x=smear_edges[:-1],
        height=s_mean_adv_ansatz_fr,
        width=WIDTH_SMEAR,
        align='edge', fill=False, edgecolor='black', label='Adversarial'
    )
    ax[0].bar(
        x=smear_edges[:-1],
        height=s_means_fr,
        width=WIDTH_SMEAR,
        align='edge', fill=False, edgecolor='red', label='True GMM', linestyle='--'
    )
    ax[0].scatter(
        (smear_edges[:-1] + smear_edges[1:]) / 2, ansatz_data[min_min_idx, :], label = 'Observations responsible for the \nAdversarial Ansatz'
    )

    # true
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=true_means_w_ansatz,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='blue', label='Misspecified GMM'
    )
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=H @ t_mean_adv_ansatz_fr,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black', label='Adversarial'
    )
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=true_means_w,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='red', label='True Intensity', linestyle='--'
    )

    # axis labels
    ax[0].set_ylabel('Expected Bin Counts')

    ax[0].legend()
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure3(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    Sample LS intervals for direct wide-bin unfolding and estimated bin-wise
    coverage.

    Parameters:
    -----------
        true_edges_wide (np arr) : edges of the wide true bins
        save_loc        (str)    : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the true wide bin data
    true_bin_means_obj = np.load(file='./bin_means/gmm_wide.npz')
    true_means_w_ansatz = true_bin_means_obj['t_means_ansatz_w']
    true_means_w = true_bin_means_obj['t_means_w']

    # read in the computed intervals
    ints_obj = np.load(file='./data/wide_bin_deconvolution/ints_cov_wide_ls.npz')
    intervals = ints_obj['intervals']
    coverage = ints_obj['coverage']

    num_sims = intervals.shape[0]

    # put the above two together
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # example intervals
    # true bin expected counts
    ax[0].bar(
        x=true_edges_wide[:-1],
        height=true_means_w,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black', label='True Bin Expected Counts'
    )

    # Recovered intervals
    interval_midpoints = (intervals[0, :, 1] + intervals[0, :, 0]) / 2
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=interval_midpoints,
        yerr=np.abs(intervals[0, :, :] - interval_midpoints[:, np.newaxis]).T,
        capsize=7, ls='none', label='Least-squares 95% Confidence Intervals', color='red'
    )

    ax[0].set_ylabel('Bin Count')
    ax[0].legend(bbox_to_anchor=(0.75, 1.15))
    ax[0].set_ylim(bottom=-100)

    # coverage
    # find the clopper-pearson intervals
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage]
    ).T

    ax[1].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage,
        yerr=np.abs(coverage - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    # plot the coverage
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=coverage,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )

    # plot the desired level
    ax[1].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    ax[1].set_ylabel('Estimated Coverage')
    ax[1].legend(bbox_to_anchor=(0.75, 1.15))

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure4(
    true_edges=np.linspace(-7, 7, 41),
    save_loc=None
):
    """
    Unfolding with n = 40 true bins and least-squares intervals.

    Parameters:
    -----------
        true_edges (np arr) : edges of the wide true bins
        save_loc   (str)    : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the true bin data
    true_bin_means_obj = np.load(file='./bin_means/gmm_fr.npz')
    true_means_ansatz = true_bin_means_obj['t_means_ansatz_fr']
    true_means = true_bin_means_obj['t_means_fr']

    # read in the computed intervals
    ints_obj = np.load(file='./data/wide_bin_deconvolution/ints_cov_fine_ls.npz')
    intervals = ints_obj['intervals']
    coverage = ints_obj['coverage']

    num_sims = intervals.shape[0]

    # put the above two together
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    WIDTH_TRUE = true_edges[1] - true_edges[0]

    # example interval
    # true bin expected counts
    ax[0].bar(
        x=true_edges[:-1],
        height=true_means,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black', label='True Bin Expected Counts'
    )

    # Recovered intervals
    interval_midpoints = (intervals[0, :, 1] + intervals[0, :, 0]) / 2
    ax[0].errorbar(
        x=(true_edges[1:] + true_edges[:-1]) / 2,
        y=interval_midpoints,
        yerr=np.abs(intervals[0, :, :] - interval_midpoints[:, np.newaxis]).T,
        capsize=4, ls='none', label='Least-squares 95% Confidence Intervals', color='red'
    )

    ax[0].set_ylabel('Bin Count')
    ax[0].legend(bbox_to_anchor=(0.75, 1.15))

    # coverage
    # find the clopper-pearson intervals
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage]
    ).T

    ax[1].errorbar(
        x=(true_edges[1:] + true_edges[:-1]) / 2,
        y=coverage,
        yerr=np.abs(coverage - clop_pears_ints),
        capsize=3, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    # plot the coverage
    ax[1].bar(
        x=true_edges[:-1],
        height=coverage,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )

    # plot the desired level
    ax[1].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    ax[1].set_ylabel('Estimated Coverage')
    ax[1].legend(bbox_to_anchor=(0.75, 1.15))

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure5(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    Unfolding with post-inversion aggregation to 10 intervals.

    Parameters:
    -----------
        true_edges_wide (np arr) : edges of the wide true bins
        save_loc        (str)    : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the true wide bin data
    true_bin_means_obj = np.load(file='./bin_means/gmm_wide.npz')
    true_means_w_ansatz = true_bin_means_obj['t_means_ansatz_w']
    true_means_w = true_bin_means_obj['t_means_w']

    # read in the computed intervals
    ints_obj = np.load(file='./data/wide_bin_deconvolution/ints_cov_agg_ls.npz')
    intervals = ints_obj['intervals']
    coverage = ints_obj['coverage']

    num_sims = intervals.shape[0]

    # put the above two together
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # example interval
    # true bin expected counts
    ax[0].bar(
        x=true_edges_wide[:-1],
        height=true_means_w,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black', label='True Bin Expected Counts'
    )

    # Recovered intervals
    INT_EX_IDX = 1 
    interval_midpoints = (intervals[INT_EX_IDX, :, 1] + intervals[INT_EX_IDX, :, 0]) / 2
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=interval_midpoints,
        yerr=np.abs(intervals[INT_EX_IDX, :, :] - interval_midpoints[:, np.newaxis]).T,
        capsize=7, ls='none', label='Least-squares 95% Confidence Intervals', color='red'
    )

    ax[0].set_ylabel('Bin Count')
    ax[0].legend(bbox_to_anchor=(0.75, 1.15))

    # coverage
    # find the clopper-pearson intervals
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage]
    ).T

    ax[1].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage,
        yerr=np.abs(coverage - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    # plot the coverage
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=coverage,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )

    # plot the desired level
    ax[1].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    ax[1].set_ylabel('Estimated Coverage')
    ax[1].legend(bbox_to_anchor=(0.75, 1.15))

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure6(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    Comparing Least-Squares, OSB, and PO intervals, computed using the smearing
    matrix generated with the GMM Ansatz.

    Parameters:
    -----------
        true_edges_wide (np arr) : edges of the wide true bins
        save_loc        (str)    : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the true wide bin data
    true_means_w = np.load(file='./bin_means/gmm_wide.npz')['t_means_w']

    # read in the computed intervals
    intervals_ls_agg = np.load(
        file='./data/wide_bin_deconvolution/ints_cov_agg_ls.npz'
    )['intervals']
    intervals_full_rank_gmm_ans_files = np.load(
        file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz.npz'
    )
    intervals_osb_fr = intervals_full_rank_gmm_ans_files['intervals_osb_fr']
    intervals_po_fr = intervals_full_rank_gmm_ans_files['intervals_po_fr']

    # compute mean/std widths
    ls_mean_widths_fr, ls_std_widths_fr = compute_mean_std_width(intervals=intervals_ls_agg)
    osb_mean_widths_fr, osb_std_widths_fr = compute_mean_std_width(intervals=intervals_osb_fr)
    po_mean_widths_fr, po_std_widths_fr = compute_mean_std_width(intervals=intervals_po_fr)

    # combining the expected interval lengths with the example
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]
    NUM_SIMS = intervals_ls_agg.shape[0]

    # single interval instance
    ax[0].bar(
        x=true_edges_wide[:-1],
        height=true_means_w,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black', label='True Bin Expected Counts'
    )

    INT_EX_IDX = 1 
    midpoints_ls = (intervals_ls_agg[INT_EX_IDX, :, 1] + intervals_ls_agg[INT_EX_IDX, :, 0]) / 2
    midpoints_osb = (intervals_osb_fr[INT_EX_IDX, :, 1] + intervals_osb_fr[INT_EX_IDX, :, 0]) / 2
    midpoints_po = (intervals_po_fr[INT_EX_IDX, :, 1] + intervals_po_fr[INT_EX_IDX, :, 0]) / 2

    # LS
    CAPSIZE = 4
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2 - (WIDTH_TRUE / 3),
        y=midpoints_ls,
        yerr=np.abs(intervals_ls_agg[INT_EX_IDX, :, :] - midpoints_ls[:, np.newaxis]).T,
        capsize=CAPSIZE, ls='none', label='Least-squares', color='black'
    )

    # OSB
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=midpoints_osb,
        yerr=np.abs(intervals_osb_fr[INT_EX_IDX, :, :] - midpoints_osb[:, np.newaxis]).T,
        capsize=CAPSIZE, ls='none', label='OSB', color='blue'
    )

    # PO
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2 + (WIDTH_TRUE / 3),
        y=midpoints_osb,
        yerr=np.abs(intervals_po_fr[INT_EX_IDX, :, :] - midpoints_po[:, np.newaxis]).T,
        capsize=CAPSIZE, ls='none', label='PO', color='red'
    )

    ax[0].set_ylabel('Bin Count')
    ax[0].legend()

    # mean widths
    # LS intervals
    x_plot = np.arange(1, 11)
    ax[1].plot(x_plot, ls_mean_widths_fr, label='Least-squares', color='black')

    ax[1].errorbar(
        x=x_plot,
        y=ls_mean_widths_fr,
        yerr=2 * ls_std_widths_fr,
        capsize=7, ls='none', color='black'
    )

    # osb intervals
    ax[1].plot(x_plot, osb_mean_widths_fr, label='OSB', color='blue')

    ax[1].errorbar(
        x=x_plot,
        y=osb_mean_widths_fr,
        yerr=2 * osb_std_widths_fr / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='blue'
    )

    # PO intervals
    ax[1].plot(x_plot, po_mean_widths_fr, label='PO', color='red')

    ax[1].errorbar(
        x=x_plot,
        y=po_mean_widths_fr,
        yerr=2 * po_std_widths_fr / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='red'
    )

    # axis labels
    ax[1].set_ylabel(r'Average Interval Length (Error bars are $\pm 2$se)')
    ax[1].set_xlabel('Bin Number')
    ax[1].set_xticks(x_plot)
    ax[1].legend()

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)
    
    plt.show()


def plot_figure7(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    OSB and PO bin-wise coverage for the GMM ansatz.

    Parameters:
    -----------
        true_edges_wide (np arr) : edges of the wide true bins
        save_loc        (str)    : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    coverage_obj = np.load(
        file='./data/wide_bin_deconvolution/coverage_osb_po_full_rank_misspec_gmm_ansatz.npz'
    )
    coverage_osb_fr = coverage_obj['coverage_osb_fr']
    coverage_po_fr = coverage_obj['coverage_po_fr']

    num_sims = 1000
    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # showing coverage of OSB and PO is preserved
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 3))

    # OSB Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage_osb_fr]
    ).T

    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_osb_fr,
        yerr=np.abs(coverage_osb_fr - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    # plot the coverage
    ax[0].bar(
        x=true_edges_wide[:-1],
        height=coverage_osb_fr,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )

    # plot the desired level
    ax[0].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')
    ax[0].set_ylabel('Estimated Coverage')

    # PO Coverage
    # find the clopper-pearson intervals
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage_po_fr]
    ).T

    ax[1].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_po_fr,
        yerr=np.abs(coverage_po_fr - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    # plot the coverage
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=coverage_po_fr,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )

    # plot the desired level
    ax[1].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    # cut off the bottom
    ax[0].set_ylim(bottom=0.6)
    ax[1].set_ylim(bottom=0.6)

    # add legend
    ax[0].legend(bbox_to_anchor=(0.25, 1.12))

    # add titles
    ax[0].set_title('OSB')
    ax[1].set_title('PO')

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()