"""
A strict to make *bespoke* plots for sundry purposes.

Author        : Michael Stanley
Created       : 26 Oct 2021
Last Modified : 26 Oct 2021
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


def plot_proposal_plot1(
    true_edges_wide=np.linspace(-7, 7, 11),
    save_loc=None
):
    """
    OSB coverage for the GMM ansatz.

    This function is made specifically for the updated OSB coverage plot in my
    proposal presentation to make sure it is of the same size as the least
    squares coverage plot.

    Based on plot_figure7 in ./plotting_deconvolution.py.

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

    num_sims = 1000
    WIDTH_TRUE = true_edges_wide[1] - true_edges_wide[0]

    # showing coverage of OSB and PO is preserved
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

    # OSB Coverage
    clop_pears_ints = np.array(
        [proportion_confint(i*num_sims, num_sims, alpha=0.05, method='beta') for i in coverage_osb_fr]
    ).T

    ax.errorbar(
        x=(true_edges_wide[1:] + true_edges_wide[:-1]) / 2,
        y=coverage_osb_fr,
        yerr=np.abs(coverage_osb_fr - clop_pears_ints),
        capsize=7, ls='none', label=r'95% Clopper-Pearson Intervals ($M_D = 1000$)'
    )

    # plot the coverage
    ax.bar(
        x=true_edges_wide[:-1],
        height=coverage_osb_fr,
        width=WIDTH_TRUE,
        align='edge', fill=False, edgecolor='black'
    )

    # plot the desired level
    ax.axhline(0.95, linestyle='--', color='red', alpha=0.6, label='Nominal Coverage Level')
    ax.set_ylabel('Estimated Coverage', fontsize='large')

    # add legend
    ax.legend(bbox_to_anchor=(1, 1.22), fontsize='x-large')

    # add titles
    # ax.set_title('OSB')

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc, dpi=300)

    plt.show()


if __name__ == "__main__":

    # define base paths
    PROPOSAL_DIR = "/Users/mikestanley/Carnegie Mellon Documents/proposal"
    PROPOSAL_DIR += "/proposal_talk/images"

    # new OSB plot
    plot_proposal_plot1(
        save_loc=PROPOSAL_DIR + "/osb_coverage.png"
    )