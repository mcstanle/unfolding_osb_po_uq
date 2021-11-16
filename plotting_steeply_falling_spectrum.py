"""
Script containing all code to generate the plots for section 5.

This script is written so that each function can plot its respective figure in
isolation from any other figure. This way, one can simply toggle the figures
one wants to plot in the main section of the code.

The convention is that the code to plot figure k can be found in function
called "plot_figure[k]". 

Author        : Michael Stanley
Created       : 16 Nov 2021
Last Modified : 16 Nov 2021
===============================================================================
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from utils import intensity_f, compute_mean_std_width

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
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
    ax3 = fig.add_subplot(1, 3, 3)

    WIDTH_REAL = unfold_grid[1] - unfold_grid[0]
    EP_DIFFS = np.diff(close_func_endpoints)

    # -- intensity functions
    grid_gev = np.linspace(400, 1000, 200)
    f_true_grid = [f_true(p_i) for p_i in grid_gev]
    f_ansatz_grid = [f_ansatz(p_i) for p_i in grid_gev]
    ax1.plot(grid_gev, f_true_grid, label='True')
    ax1.plot(grid_gev, f_ansatz_grid, label='Ansatz')

    # -- base10 plots
    # truth
    ax2.bar(
        x=unfold_grid[:-1],
        height=unfold_means / WIDTH_REAL,
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
        x=unfold_grid[:-1],
        height=np.log(unfold_means / WIDTH_REAL),
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
    # plt.savefig(SAVE_PATH + '/steeply_falling_rank_deficient_uneven_bins/steep_fall_intensity_and_bin.png', dpi=300)
    plt.show()