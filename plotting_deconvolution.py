"""
Script containing all code to generate the plots for section 4.

This script is written so that each function can plot its respective figure in
isolation from any other figure. This way, one can simply toggle the figures
one wants to plot in the main section of the code.

The convention is that the code to plot figure k can be found in function
called "plot_figure[k]". 

Author        : Michael Stanley
Created       : 04 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
from compute_K_deconvolution_adversarial_ansatz import compute_adversarial_ansatz
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
    plt.ylabel('Intensity', fontsize='x-large')
    plt.xlabel('Physical Observable', fontsize='x-large')

    # tick sizes
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.legend(fontsize='x-large')
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
        (smear_edges[:-1] + smear_edges[1:]) / 2, ansatz_data[min_min_idx, :], label = 'Adversarial Ansatz\nObservations'
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
    ax[0].set_ylabel('Expected Bin Counts', fontsize='x-large')

    # increase the axis tick sizes
    ax[0].tick_params(axis='x', labelsize='x-large')
    ax[1].tick_params(axis='x', labelsize='x-large')
    ax[0].tick_params(axis='y', labelsize='x-large')
    ax[1].tick_params(axis='y', labelsize='x-large')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax[0].legend(fontsize='x-large')
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

    ax[0].set_ylabel('Bin Count', fontsize='x-large')
    ax[0].legend(bbox_to_anchor=(1, 1.22), fontsize='x-large')
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

    ax[1].set_ylabel('Estimated Coverage', fontsize='x-large')
    ax[1].legend(bbox_to_anchor=(1, 1.22), fontsize='x-large')

    # axis label size adjustments
    ax[0].tick_params(axis='x', labelsize='x-large')
    ax[1].tick_params(axis='x', labelsize='x-large')
    ax[0].tick_params(axis='y', labelsize='x-large')
    ax[1].tick_params(axis='y', labelsize='x-large')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

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

    ax[0].set_ylabel('Bin Count', fontsize='x-large')
    ax[0].legend(bbox_to_anchor=(0.85, 1.2), fontsize='large')

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

    ax[1].set_ylabel('Estimated Coverage', fontsize='x-large')
    ax[1].legend(bbox_to_anchor=(0.95, 1.2), fontsize='large')

    # adjust tick label sizes
    ax[0].tick_params(axis='x', labelsize='x-large')
    ax[1].tick_params(axis='x', labelsize='x-large')
    ax[0].tick_params(axis='y', labelsize='x-large')
    ax[1].tick_params(axis='y', labelsize='x-large')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

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

    ax[0].set_ylabel('Bin Count', fontsize='x-large')
    ax[0].legend(bbox_to_anchor=(0.85, 1.18), fontsize='large')

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

    ax[1].set_ylabel('Estimated Coverage', fontsize='x-large')
    ax[1].legend(bbox_to_anchor=(0.85, 1.18), fontsize='large')

    # adjust tick label sizes
    ax[0].tick_params(axis='x', labelsize='x-large')
    ax[1].tick_params(axis='x', labelsize='x-large')
    ax[0].tick_params(axis='y', labelsize='x-large')
    ax[1].tick_params(axis='y', labelsize='x-large')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

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
        file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz_ORIGINAL.npz'
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

    # put the scales in scientific notation
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

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
    ax[0].set_ylabel('Estimated Coverage', fontsize='large')

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
    ax[0].legend(bbox_to_anchor=(0.2, 1.12), fontsize='large')

    # add titles
    ax[0].set_title('OSB')
    ax[1].set_title('PO')

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure8(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    OSB and PO bin-wise coverage for the GMM adversarial ansatz.

    Parameters:
    -----------
        true_edges_wide (np arr) : edges of the wide true bins
        save_loc        (str)    : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # least-squares Coverage
    coverage_ls_adv = np.load(file='./data/wide_bin_deconvolution/ints_cov_agg_adv_ls.npz')['coverage']

    # OSB and PO Coverage
    coverage_obj = np.load(
        file='./data/wide_bin_deconvolution/coverage_osb_po_full_rank_adv_ansatz.npz'
    )

    coverage_osb_adv = coverage_obj['coverage_osb_adv']
    coverage_po_adv = coverage_obj['coverage_po_adv']

    # showing coverage of LS/OSB/PO Intervals
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
    NUM_SIMS = 1000
    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # LS Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*NUM_SIMS, NUM_SIMS, alpha=0.05, method='beta') for i in coverage_ls_adv]
    ).T
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_ls_adv,
        yerr=np.abs(coverage_ls_adv - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )
    ax[0].bar(
        x=true_edges_wide[:-1],
        height=coverage_ls_adv,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )
    ax[0].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')
    ax[0].set_ylabel('Estimated Coverage', fontsize='large')

    # OSB Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*NUM_SIMS, NUM_SIMS, alpha=0.05, method='beta') for i in coverage_osb_adv]
    ).T
    ax[1].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_osb_adv,
        yerr=np.abs(coverage_osb_adv - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($N = 1000$)'
    )
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=coverage_osb_adv,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )
    ax[1].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')


    # PO Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*NUM_SIMS, NUM_SIMS, alpha=0.05, method='beta') for i in coverage_po_adv]
    ).T
    ax[2].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_po_adv,
        yerr=np.abs(coverage_po_adv - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($N = 1000$)'
    )
    ax[2].bar(
        x=true_edges_wide[:-1],
        height=coverage_po_adv,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )
    ax[2].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    # lop off the bottom
    ax[0].set_ylim(bottom=0.6)
    ax[1].set_ylim(bottom=0.6)
    ax[2].set_ylim(bottom=0.6)

    # legend
    ax[0].legend(bbox_to_anchor=(1.2, 1.35))

    # titles
    ax[0].set_title('LS')
    ax[1].set_title('OSB')
    ax[2].set_title('PO')

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure9(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    OSB and PO bin-wise coverage for the adversarial ansatz, but computed with
    80 true bins to minimize systematic error.

    Parameters:
    -----------
        true_edges_wide (np arr) : edges of the wide true bins
        save_loc        (str)    : saving location -- not saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    coverage_obj = np.load(
        file='./data/wide_bin_deconvolution/coverage_osb_po_rank_def_adv_ansatz.npz'
    )
    coverage_osb_adv_80 = coverage_obj['coverage_osb_adv_80']
    coverage_po_adv_80 = coverage_obj['coverage_po_adv_80']

    NUM_SIMS = 1000
    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # showing coverage of LS/OSB/PO Intervals
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 3), sharey=False)

    # OSB Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*NUM_SIMS, NUM_SIMS, alpha=0.05, method='beta') for i in coverage_osb_adv_80]
    ).T
    ax[0].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_osb_adv_80,
        yerr=np.abs(coverage_osb_adv_80 - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )
    ax[0].bar(
        x=true_edges_wide[:-1],
        height=coverage_osb_adv_80,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )
    ax[0].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    # PO Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*NUM_SIMS, NUM_SIMS, alpha=0.05, method='beta') for i in coverage_po_adv_80]
    ).T
    ax[1].errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_po_adv_80,
        yerr=np.abs(coverage_po_adv_80 - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($N = 1000$)'
    )
    ax[1].bar(
        x=true_edges_wide[:-1],
        height=coverage_po_adv_80,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )
    ax[1].axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    # cut off the bottom
    ax[0].set_ylim(bottom=0.6)
    ax[1].set_ylim(bottom=0.6)

    # add legend
    ax[0].legend(bbox_to_anchor=(0.25, 1.12))

    # add titles
    ax[0].set_title('OSB')
    ax[1].set_title('PO')

    # axis label
    ax[0].set_ylabel('Estimated Coverage')

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure10(save_loc=None):
    """
    Expected Intervals Widths across LS, OSB, PO, and matrix rank.

    Parameters:
    -----------
        save_loc        (str)    : saving location -- not saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the computed intervals
    intervals_ls_agg = np.load(
        file='./data/wide_bin_deconvolution/ints_cov_agg_ls.npz'
    )['intervals']
    intervals_files_fr = np.load(
        file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz_ORIGINAL.npz'
    )
    intervals_osb_fr = intervals_files_fr['intervals_osb_fr']
    intervals_po_fr = intervals_files_fr['intervals_po_fr']

    intervals_files_rd = np.load(
        file='./data/wide_bin_deconvolution/intervals_osb_po_rank_def_adversarial_ansatz_ORIGINAL.npz'
    )
    intervals_osb_adv_80 = intervals_files_rd['intervals_osb_adv_80']
    intervals_po_adv_80 = intervals_files_rd['intervals_po_adv_80']

    # compute mean/std widths
    ls_mean_widths_fr, ls_std_widths_fr = compute_mean_std_width(intervals=intervals_ls_agg)
    osb_mean_widths_fr, osb_std_widths_fr = compute_mean_std_width(intervals=intervals_osb_fr)
    po_mean_widths_fr, po_std_widths_fr = compute_mean_std_width(intervals=intervals_po_fr)
    osb_mean_widths_rd, osb_std_widths_rd = compute_mean_std_width(intervals=intervals_osb_adv_80)
    po_mean_widths_rd, po_std_widths_rd = compute_mean_std_width(intervals=intervals_po_adv_80)

    # look at interval expected length
    plt.figure(figsize=(10,5))

    x_plot = np.arange(1, 11)

    # LS intervals
    plt.plot(
        x_plot, ls_mean_widths_fr,
        label='Least-squares (40 True Bins)', color='black', linestyle='--', alpha=0.65
    )

    # osb intervals
    plt.plot(
        x_plot, osb_mean_widths_fr,
        label='OSB (40 True Bins)', color='blue', linestyle='--', alpha=0.65
    )
    plt.plot(
        x_plot, osb_mean_widths_rd,
        label='OSB (80 True Bins)', color='blue', alpha=0.95
    )

    # PO intervals
    plt.plot(
        x_plot, po_mean_widths_fr,
        label='PO (40 True Bins)', color='red', linestyle='--', alpha=0.65
    )
    plt.plot(
        x_plot, po_mean_widths_rd,
        label='PO (80 True Bins)', color='red', alpha=0.95
    )

    # axis labels
    plt.ylabel(r'Average Interval Length')
    plt.xlabel('Bin Number')

    plt.xticks(x_plot)

    # change yscale to scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.legend()
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure11(save_loc=None):
    """
    Expected Intervals Widths across LS, OSB, PO, SSB, and Minimax bounds

    Parameters:
    -----------
        save_loc        (str)    : saving location -- not saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # least-squares
    intervals_ls_agg = np.load(
        file='./data/wide_bin_deconvolution/ints_cov_agg_ls.npz'
    )['intervals']
    ls_mean_widths_fr, ls_std_widths_fr = compute_mean_std_width(intervals=intervals_ls_agg)

    # OSB & PO
    intervals_osb_po_fr = np.load(
        file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz_ORIGINAL.npz'
    )
    intervals_osb_fr = intervals_osb_po_fr['intervals_osb_fr']
    intervals_po_fr = intervals_osb_po_fr['intervals_po_fr']
    osb_mean_widths_fr, osb_std_widths_fr = compute_mean_std_width(intervals=intervals_osb_fr)
    po_mean_widths_fr, po_std_widths_fr = compute_mean_std_width(intervals=intervals_po_fr)

    # SSB and Minimax
    ssb_mm_obj = np.load(
        file='./data/wide_bin_deconvolution/intervals_ssb_minimax_full_rank_misspec_gmm_ansatz.npz'
    )
    minimax_interval_widths = ssb_mm_obj['minimax_interval_widths']
    ssb_intervals = ssb_mm_obj['ssb_intervals']
    ssb_mean_widths, ssb_std_widths = compute_mean_std_width(intervals=ssb_intervals)

    # look at interval expected length
    plt.figure(figsize=(10,5))

    NUM_SIMS = 1000
    x_plot = np.arange(1, 11)
    gauss_quant = stats.norm.ppf(97.5)

    # LS intervals
    plt.plot(x_plot, ls_mean_widths_fr, label='Least-squares', color='black')

    plt.errorbar(
        x=x_plot,
        y=ls_mean_widths_fr,
        yerr=2 * ls_std_widths_fr,
        capsize=7, ls='none', color='black'
    )

    # osb intervals
    plt.plot(x_plot, osb_mean_widths_fr, label='OSB', color='blue')

    plt.errorbar(
        x=x_plot,
        y=osb_mean_widths_fr,
        yerr=2 * osb_std_widths_fr / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='blue'
    )

    # PO intervals
    plt.plot(x_plot, po_mean_widths_fr, label='PO', color='red')

    plt.errorbar(
        x=x_plot,
        y=po_mean_widths_fr,
        yerr=2 * po_std_widths_fr / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='red'
    )

    # SSB
    plt.plot(x_plot, ssb_mean_widths, label='SSB', color='purple')

    plt.errorbar(
        x=x_plot,
        y=ssb_mean_widths,
        yerr=2 * ssb_std_widths / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='purple'
    )

    # Minimax
    plt.plot(x_plot, minimax_interval_widths[:, 0] * 2, color='gray')
    plt.plot(x_plot, minimax_interval_widths[:, 1] * 2, color='gray')
    plt.fill_between(
        x_plot, minimax_interval_widths[:, 0] * 2, minimax_interval_widths[:, 1] * 2,
        color='gray',
        alpha=0.2,
        label='Minimax Range'
    )

    # axis labels
    plt.ylabel(r'Average Interval Length (Error bars are $\pm 2 \hat{\sigma}$)')
    plt.xlabel('Bin Number')

    plt.xticks(x_plot)

    # change yscale to scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.legend()
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure12(save_loc=None):
    """
    Expected Intervals widths across OSB and PO setups and rank deficiency of
    smearing matrix.

    Parameters:
    -----------
        save_loc        (str)    : saving location -- not saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # load the intervals
    interval_obj = np.load(file='./data/wide_bin_deconvolution/coverage_and_intervals_40_80_160_320.npz')

    intervals_osb_40 = interval_obj['intervals_osb_40']
    intervals_osb_80 = interval_obj['intervals_osb_80']
    intervals_osb_160 = interval_obj['intervals_osb_160']
    intervals_osb_320 = interval_obj['intervals_osb_320']
    intervals_po_40 = interval_obj['intervals_po_40']
    intervals_po_80 = interval_obj['intervals_po_80']
    intervals_po_160 = interval_obj['intervals_po_160']
    intervals_po_320 = interval_obj['intervals_po_320']

    # find expected widths
    osb_mean_width_40 = compute_mean_std_width(intervals=intervals_osb_40)[0]
    osb_mean_width_80 = compute_mean_std_width(intervals=intervals_osb_80)[0]
    osb_mean_width_160 = compute_mean_std_width(intervals=intervals_osb_160)[0]
    osb_mean_width_320 = compute_mean_std_width(intervals=intervals_osb_320)[0]
    po_mean_width_40 = compute_mean_std_width(intervals=intervals_po_40)[0]
    po_mean_width_80 = compute_mean_std_width(intervals=intervals_po_80)[0]
    po_mean_width_160 = compute_mean_std_width(intervals=intervals_po_160)[0]
    po_mean_width_320 = compute_mean_std_width(intervals=intervals_po_320)[0]

    # looking at length by bin
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)

    x_vals = np.arange(1, 11)

    # osb
    ax[0].plot(x_vals, osb_mean_width_40, label='40 Bins')
    ax[0].plot(x_vals, osb_mean_width_80, label='80 Bins')
    ax[0].plot(x_vals, osb_mean_width_160, label='160 Bins')
    ax[0].plot(x_vals, osb_mean_width_320, label='320 Bins')

    # po
    ax[1].plot(x_vals, po_mean_width_40)
    ax[1].plot(x_vals, po_mean_width_80)
    ax[1].plot(x_vals, po_mean_width_160)
    ax[1].plot(x_vals, po_mean_width_320)

    # labels
    ax[0].set_ylabel('Expected Width')
    ax[0].set_xlabel('Bin #')
    ax[1].set_xlabel('Bin #')

    ax[0].legend()

    plt.tight_layout()
    
    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure13(save_loc=None):
    """
    Looking at the sensitivity of the expected interval width with respect to
    the rank deficiency

    Parameters:
    -----------
        save_loc        (str)    : saving location -- not saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # load the intervals
    interval_obj = np.load(file='./data/wide_bin_deconvolution/coverage_and_intervals_40_80_160_320.npz')

    intervals_osb_40 = interval_obj['intervals_osb_40']
    intervals_osb_80 = interval_obj['intervals_osb_80']
    intervals_osb_160 = interval_obj['intervals_osb_160']
    intervals_osb_320 = interval_obj['intervals_osb_320']
    intervals_po_40 = interval_obj['intervals_po_40']
    intervals_po_80 = interval_obj['intervals_po_80']
    intervals_po_160 = interval_obj['intervals_po_160']
    intervals_po_320 = interval_obj['intervals_po_320']

    # find expected widths
    osb_mean_width_40 = compute_mean_std_width(intervals=intervals_osb_40)[0]
    osb_mean_width_80 = compute_mean_std_width(intervals=intervals_osb_80)[0]
    osb_mean_width_160 = compute_mean_std_width(intervals=intervals_osb_160)[0]
    osb_mean_width_320 = compute_mean_std_width(intervals=intervals_osb_320)[0]
    po_mean_width_40 = compute_mean_std_width(intervals=intervals_po_40)[0]
    po_mean_width_80 = compute_mean_std_width(intervals=intervals_po_80)[0]
    po_mean_width_160 = compute_mean_std_width(intervals=intervals_po_160)[0]
    po_mean_width_320 = compute_mean_std_width(intervals=intervals_po_320)[0]

    # stack the above for each of plotting
    osb_mean_widths = np.vstack((
        osb_mean_width_40,
        osb_mean_width_80,
        osb_mean_width_160,
        osb_mean_width_320,
    ))
    po_mean_widths = np.vstack((
        po_mean_width_40,
        po_mean_width_80,
        po_mean_width_160,
        po_mean_width_320,
    ))

    # look at bin interval length trajectories
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)

    x_vals = [40, 80, 160, 320]
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']

    # osb intervals
    for i in range(10):
        ax[0].plot(x_vals, osb_mean_widths[:, i], label='Bin %i' % (i + 1), linestyle=linestyles[i % 4])

    # po intervals
    for i in range(10):
        ax[1].plot(x_vals, po_mean_widths[:, i], label='Bin %i' % (i + 1), linestyle=linestyles[i % 4])

    ax[0].legend(bbox_to_anchor=(1, .7), fontsize='large')

    # set bin count axis
    ax[0].set_xticks(x_vals)
    ax[0].set_xticklabels(x_vals)
    ax[1].set_xticks(x_vals)
    ax[1].set_xticklabels(x_vals)

    # ax[0].set_xlabel('Unfold Dimension', fontsize='x-large')
    # ax[1].set_xlabel('Unfold Dimension', fontsize='x-large')
    fig.text(0.5, 0.0, r'Unfold Dimension', fontsize='x-large', ha='center')

    ax[0].set_ylabel('Expected Width', fontsize='x-large')

    # adjust tick sizes
    ax[0].tick_params(axis='x', labelsize='x-large')
    ax[1].tick_params(axis='x', labelsize='x-large')
    ax[0].tick_params(axis='y', labelsize='x-large')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure14(save_loc=None):
    """
    Sensitivity of PO interval widths to choice of prior

    Parameters:
    -----------
        save_loc        (str)    : saving location -- not saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    load_objects = np.load(
        file='./data/wide_bin_deconvolution/po_intervals_flat_ansatz_correct_ORIGINAL.npz'
    )

    intervals_po_flat = load_objects['intervals_po_flat']
    intervals_po_ansatz = load_objects['intervals_po_ansatz']
    intervals_po_correct = load_objects['intervals_po_correct']

    # PO Adversarial Intervals
    intervals_files = np.load(
        file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_adv_ansatz_ORIGINAL.npz'
    )
    intervals_po_adv = intervals_files['intervals_po_adv']

    # OSB with GMM ansatz
    intervals_osb = np.load(
        file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz_ORIGINAL.npz'
    )['intervals_osb_fr']

    # mean intervals widths
    po_mean_widths_flat, po_std_widths_flat = compute_mean_std_width(intervals=intervals_po_flat)
    po_mean_widths_ansatz, po_std_widths_ansatz = compute_mean_std_width(intervals=intervals_po_ansatz)
    po_mean_widths_ansatz_adv, po_std_widths_ansatz_adv = compute_mean_std_width(intervals=intervals_po_adv)
    po_mean_widths_correct, po_std_widths_correct = compute_mean_std_width(intervals=intervals_po_correct)
    osb_mean_widths, osb_std_widths = compute_mean_std_width(intervals=intervals_osb)

    # look at interval expected length
    plt.figure(figsize=(10,5))

    NUM_SIMS = 1000
    ALPHA = 0.05

    x_plot = np.arange(1, 11)
    gauss_quant = stats.norm.ppf(97.5)

    # Flat
    plt.plot(x_plot, po_mean_widths_flat, label='Flat', color='black')

    plt.errorbar(
        x=x_plot,
        y=po_mean_widths_flat,
        yerr=2 * po_std_widths_flat / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='black'
    )

    # Ansatz
    plt.plot(x_plot, po_mean_widths_ansatz, label='Misspecified GMM', color='blue')

    plt.errorbar(
        x=x_plot,
        y=po_mean_widths_ansatz,
        yerr=2 * po_std_widths_ansatz / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='blue'
    )

    # adversarial
    plt.plot(x_plot, po_mean_widths_ansatz_adv, label='Adversarial', color='green')

    plt.errorbar(
        x=x_plot,
        y=po_mean_widths_ansatz_adv,
        yerr=2 * po_std_widths_ansatz_adv / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='green'
    )

    # Correct
    plt.plot(x_plot, po_mean_widths_correct, label='Correct', color='red')

    plt.errorbar(
        x=x_plot,
        y=po_mean_widths_correct,
        yerr=2 * po_std_widths_correct / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='red'
    )

    # OSB
    plt.plot(x_plot, osb_mean_widths, label='OSB', color='gray')

    plt.errorbar(
        x=x_plot,
        y=osb_mean_widths,
        yerr=2 * osb_std_widths / np.sqrt(NUM_SIMS),
        capsize=7, ls='none', color='gray'
    )

    # axis labels
    plt.ylabel(r'Average Interval Length (Error bars $\pm 2$se)', fontsize='x-large')
    plt.xlabel('Bin Number', fontsize='x-large')

    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.legend(fontsize='x-large')
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()