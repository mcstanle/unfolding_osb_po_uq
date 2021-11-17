"""
Script containing all code to generate the plots for section 5.

This script is written so that each function can plot its respective figure in
isolation from any other figure. This way, one can simply toggle the figures
one wants to plot in the main section of the code.

The convention is that the code to plot figure k can be found in function
called "plot_figure[k]". 

Author        : Michael Stanley
Created       : 16 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
import json
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from steeply_falling_spectra import compute_intensities

plt.style.use('seaborn-colorblind')


def plot_figure15(save_loc=None):
    """
    Shows the true and ansatz intensity functions and their binned 

    Parameters:
    -----------
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # define grids
    true_grid = np.linspace(start=400, stop=1000, num=60 + 1)
    H = np.load(
        file='./functionals/H_steeply_falling_spectrum.npy'
    )

    # read in the endpoints
    close_func_endpoints = np.load('./functionals/sfs_close_func_endpoints.npy')

    # read in the bin means
    t_means = np.load(
        file='./bin_means/sfs_rd.npz'
    )['t_means']

    # compute the true functional values
    true_func_vals = H @ t_means

    # find the intensities
    f_true, f_ans = compute_intensities()

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
    ax3 = fig.add_subplot(1, 3, 3)

    WIDTH_REAL = true_grid[1] - true_grid[0]
    EP_DIFFS = np.diff(close_func_endpoints)

    # -- intensity functions
    grid_gev = np.linspace(400, 1000, 200)
    f_true_grid = [f_true(p_i) for p_i in grid_gev]
    f_ansatz_grid = [f_ans(p_i) for p_i in grid_gev]
    ax1.plot(grid_gev, f_true_grid, label='True')
    ax1.plot(grid_gev, f_ansatz_grid, label='Ansatz')

    # -- base10 plots
    # truth
    ax2.bar(
        x=true_grid[:-1],
        height=t_means / WIDTH_REAL,
        width=WIDTH_REAL,
        align='edge', fill=False, edgecolor='black', label='True Bin Means'
    )

    # agg bins
    ax2.bar(
        x=close_func_endpoints[:-1],
        height=true_func_vals / EP_DIFFS,
        width=EP_DIFFS,
        align='edge', fill=False, edgecolor='red', label='Aggregated Bin Means'
    )

    # -- log plots
    # truth
    ax3.bar(
        x=true_grid[:-1],
        height=np.log(t_means / WIDTH_REAL),
        width=WIDTH_REAL,
        align='edge', fill=False, edgecolor='black', label='True Bin Means'
    )

    # agg bins
    ax3.bar(
        x=close_func_endpoints[:-1],
        height=np.log(true_func_vals / EP_DIFFS),
        width=EP_DIFFS,
        align='edge', fill=False, edgecolor='red', label='Functional Bin Counts'
    )

    ax1.set_ylabel('Intensity')
    ax3.set_ylabel(r'$\log($Intensity$)$')
    fig.text(0.5, 0.005, r'Transverse Momentum, $p_T$ (GeV)', ha='center')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure16(save_loc=None):
    """
    Shows the component-wise discrepancy between the true and monte carlo
    ansatz smearing matrices. 

    Parameters:
    -----------
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 6]}
    )

    # read in the matrices
    mats_tall = np.load(file='./smearing_matrices/K_wide_mats_sfs.npz')
    mats = np.load(file='./smearing_matrices/K_rank_def_mats_sfs.npz')
    K_tall = mats_tall['K_wide']
    K_tall_mc = mats_tall['K_wide_mc']
    K = mats['K']
    K_mc = mats['K_mc']

    # options
    CMAP = 'bone'

    diff_30x10 = np.abs(K_tall - K_tall_mc)
    diff_30x60 = np.abs(K - K_mc)

    # set zeros to small number
    EPS = 10e-18
    diff_30x10[diff_30x10 < EPS] = EPS
    diff_30x60[diff_30x60 < EPS] = EPS

    # find the max/min diffs to normalize the colormaps
    max_val = np.max(np.concatenate((diff_30x10, diff_30x60), axis=1).flatten())
    min_val = np.min(np.concatenate((diff_30x10, diff_30x60), axis=1).flatten())

    # 30x10
    heatmap_30x10 = ax[0].imshow(
        diff_30x10, vmin=min_val, vmax=max_val, cmap=CMAP, interpolation='nearest', norm=LogNorm()
    )

    # 30x60
    heatmap_30x60 = ax[1].imshow(
        diff_30x60, vmin=min_val, vmax=max_val, cmap=CMAP, interpolation='nearest', norm=LogNorm()
    )

    # move x-axis labels to the top of the image
    ax[0].xaxis.tick_top()
    ax[1].xaxis.tick_top()

    # add colorbar to the 30x60 plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap_30x60, cax=cbar_ax)

    # add some labels
    ax[0].set_title('30x10')
    ax[1].set_title('30x60')

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure17(save_loc=None):
    """
    Plotting under-coverage of least-squares wide-bin intervals.

    Parameters:
    -----------
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the coverage data
    coverage_ls_wb = np.load(
        file='./data/steeply_falling_spectrum/ints_cov_wide_ls.npz'
    )['coverage_ls_wb']
    coverage_ls_wb_sim = coverage_ls_wb.mean(axis=0)

    num_sims = coverage_ls_wb.shape[0]

    # read in the endpoints
    close_func_endpoints = np.load('./functionals/sfs_close_func_endpoints.npy')

    # coverage
    # find the clopper-pearson intervals
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage_ls_wb_sim]
    ).T

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    WIDTH_UNFOLD = np.diff(close_func_endpoints)

    # plot the coverage
    ax.bar(
        x=close_func_endpoints[:-1],
        height=coverage_ls_wb_sim,
        width=WIDTH_UNFOLD,
        align='edge', fill=False, edgecolor='black'
    )

    ax.errorbar(
        x=(close_func_endpoints[1:] + close_func_endpoints[:-1]) / 2,
        y=coverage_ls_wb_sim,
        yerr=np.abs(coverage_ls_wb_sim - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    ax.axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')

    # move the legend
    ax.legend(bbox_to_anchor=(0.40, 1.20))

    ax.set_ylabel('Estimated Coverage')
    ax.set_xlabel('Transverse Momentum (GeV)')

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


def plot_figure18(save_loc=None):
    """
    Plotting sample 95% confidence intervals for OSB, PO, and SSB intervals.

    Parameters:
    -----------
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """
    # read in the optimized OSB/PO/SSB intervals
    intervals = np.load(
        file='./data/steeply_falling_spectrum/intervals_optimized_ansatz_rank_def_uneven_bin.npy'
    )

    # read in the least-squares intervals
    intervals_ls_wb = np.load(file='./data/steeply_falling_spectrum/ints_cov_wide_ls.npz')['intervals_ls_wb']

    # create the true functional values
    H = np.load(file='./functionals/H_steeply_falling_spectrum.npy')
    true_means = np.load(file='./bin_means/sfs_rd.npz')['t_means']
    true_func_vals = H @ true_means

    # define dictionary for intervals
    interval_dict = {
        'osb|n': intervals[0, 0, :, :, :].copy(),
        'osb|nd': intervals[0, 1, :, :, :].copy(),
        'osb|ndc': intervals[0, 2, :, :, :].copy(),
        'po|n': intervals[1, 0, :, :, :].copy(),
        'po|nd': intervals[1, 1, :, :, :].copy(),
        'po|ndc': intervals[1, 2, :, :, :].copy(),
        'ssb|n': intervals[2, 0, :, :, :].copy(),
        'ssb|nd': intervals[2, 1, :, :, :].copy(),
        'ssb|ndc': intervals[2, 2, :, :, :].copy(),
    }

    # read in the endpoints
    close_func_endpoints = np.load('./functionals/sfs_close_func_endpoints.npy')
    EP_DIFFS = np.diff(close_func_endpoints)

    WIDTH_WIDE = EP_DIFFS
    bar_position_left = (close_func_endpoints[1:] + close_func_endpoints[:-1]) / 2 - (WIDTH_WIDE/3)
    bar_position_mid = (close_func_endpoints[1:] + close_func_endpoints[:-1]) / 2
    bar_position_right = (close_func_endpoints[1:] + close_func_endpoints[:-1]) / 2 + (WIDTH_WIDE/3)

    sample_j = 100

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharex=ax1)
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax2)
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax2)

    # non-negative
    osb_midpoints_n = (interval_dict['osb|n'][:, sample_j, 1] + interval_dict['osb|n'][:, sample_j, 0]) / 2
    po_midpoints_n = (interval_dict['po|n'][:, sample_j, 1] + interval_dict['po|n'][:, sample_j, 0]) / 2
    ssb_midpoints_n = (interval_dict['ssb|n'][:, sample_j, 1] + interval_dict['ssb|n'][:, sample_j, 0]) / 2

    # non-negative/decreasing
    osb_midpoints_nd = (interval_dict['osb|nd'][:, sample_j, 1] + interval_dict['osb|nd'][:, sample_j, 0]) / 2
    po_midpoints_nd = (interval_dict['po|nd'][:, sample_j, 1] + interval_dict['po|nd'][:, sample_j, 0]) / 2
    ssb_midpoints_nd = (interval_dict['ssb|nd'][:, sample_j, 1] + interval_dict['ssb|nd'][:, sample_j, 0]) / 2

    # non-negative/decreasing/convex
    osb_midpoints_ndc = (interval_dict['osb|ndc'][:, sample_j, 1] + interval_dict['osb|ndc'][:, sample_j, 0]) / 2
    po_midpoints_ndc = (interval_dict['po|ndc'][:, sample_j, 1] + interval_dict['po|ndc'][:, sample_j, 0]) / 2
    ssb_midpoints_ndc = (interval_dict['ssb|ndc'][:, sample_j, 1] + interval_dict['ssb|ndc'][:, sample_j, 0]) / 2

    # least squares
    ls_midpoints = (intervals_ls_wb[sample_j, :, 1] + intervals_ls_wb[sample_j, :, 0])

    # non-negative
    ax1.bar(
        x=close_func_endpoints[:-1], height=true_func_vals, width=WIDTH_WIDE,
        align='edge', fill=False, edgecolor='gray', label='Aggregated Bin Means'
    )
    ax1.errorbar(
        x=bar_position_left, y=osb_midpoints_n,
        yerr=np.abs(interval_dict['osb|n'][:, sample_j, :] - osb_midpoints_n[:, np.newaxis]).T,
        capsize=4, ls='none', label='OSB', color='red'
    )
    ax1.errorbar(
        x=bar_position_mid, y=po_midpoints_n,
        yerr=np.abs(interval_dict['po|n'][:, sample_j, :] - po_midpoints_n[:, np.newaxis]).T,
        capsize=4, ls='none', label='PO', color='blue'
    )
    ax1.errorbar(
        x=bar_position_right, y=ssb_midpoints_n,
        yerr=np.abs(interval_dict['ssb|n'][:, sample_j, :] - ssb_midpoints_n[:, np.newaxis]).T,
        capsize=4, capthick=1.5, ls='none', label='SSB', color='orange'
    )

    # NN + Decr
    ax2.bar(
        x=close_func_endpoints[:-1], height=true_func_vals, width=WIDTH_WIDE,
        align='edge', fill=False, edgecolor='gray', label='Aggregated Bin Means'
    )
    ax2.errorbar(
        x=bar_position_left, y=osb_midpoints_nd,
        yerr=np.abs(interval_dict['osb|nd'][:, sample_j, :] - osb_midpoints_nd[:, np.newaxis]).T,
        capsize=4, ls='none', label='OSB', color='red'
    )
    ax2.errorbar(
        x=bar_position_mid, y=po_midpoints_nd,
        yerr=np.abs(interval_dict['po|nd'][:, sample_j, :] - po_midpoints_nd[:, np.newaxis]).T,
        capsize=4, ls='none', label='PO', color='blue'
    )
    ax2.errorbar(
        x=bar_position_right, y=ssb_midpoints_nd,
        yerr=np.abs(interval_dict['ssb|nd'][:, sample_j, :] - ssb_midpoints_nd[:, np.newaxis]).T,
        capsize=4, capthick=1.5, ls='none', label='SSB', color='orange'
    )

    # NN + decr + convx
    ax3.bar(
        x=close_func_endpoints[:-1], height=true_func_vals, width=WIDTH_WIDE,
        align='edge', fill=False, edgecolor='gray', label='Aggregated Bin Means'
    )
    ax3.errorbar(
        x=bar_position_left, y=osb_midpoints_ndc,
        yerr=np.abs(interval_dict['osb|ndc'][:, sample_j, :] - osb_midpoints_ndc[:, np.newaxis]).T,
        capsize=4, ls='none', label='OSB', color='red'
    )
    ax3.errorbar(
        x=bar_position_mid, y=po_midpoints_ndc,
        yerr=np.abs(interval_dict['po|ndc'][:, sample_j, :] - po_midpoints_ndc[:, np.newaxis]).T,
        capsize=4, ls='none', label='PO', color='blue'
    )
    ax3.errorbar(
        x=bar_position_right, y=ssb_midpoints_ndc,
        yerr=np.abs(interval_dict['ssb|ndc'][:, sample_j, :] - ssb_midpoints_ndc[:, np.newaxis]).T,
        capsize=4, capthick=1.5, ls='none', label='SSB', color='orange'
    )

    # least squares
    ls_midpoints = (interval_dict['po|ndc'][:, sample_j, 1] + interval_dict['po|ndc'][:, sample_j, 0]) / 2

    ax4.bar(
        x=close_func_endpoints[:-1], height=true_func_vals, width=WIDTH_WIDE,
        align='edge', fill=False, edgecolor='gray'
    )
    ax4.errorbar(
        x=bar_position_mid, y=ls_midpoints,
        yerr=np.abs(intervals_ls_wb[sample_j, :, :] - ls_midpoints[:, np.newaxis]).T,
        capsize=4, capthick=2, ls='none', label='Least-Squares', color='black'
    )

    # add a common axis labels
    fig.text(0, 0.5, 'Bin Counts', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 0, 'Transverse Momentum (GeV)', ha='center', fontsize=12)

    # add subtitles
    ax1.set_title('Non-Negative')
    ax2.set_title('Non-Negative/Decreasing')
    ax3.set_title('Non-Negative/Decreasing/Convex')
    ax4.set_title('Least-Squares - No Constraints')

    # set bounds for ax2
    ax2.set_ylim(350, 1e6)

    # switch y-axis to log scale
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')

    # legends
    ax2.legend()
    ax4.legend()

    # shut off the bottom right axis
    # ax[1, 1].set_axis_off()

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()