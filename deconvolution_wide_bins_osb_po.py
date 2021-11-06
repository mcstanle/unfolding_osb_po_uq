"""
Executes the simulation studies of the OSB and PO intervals under a variety
of settings.

There are two considered settings:
1. GMM Ansatz
2. Adversarial Ansatz

Author        : Michael Stanley
Created       : 06 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
import numpy as np
from utils import compute_coverage

if __name__ == "__main__":

    # operational switches
    GMM_ANSATZ = True
    ADVERSARIAL_ANSATZ = False
    READ_INTERVALS = True

    # read in the true aggregated bin means
    t_means_w = np.load(file='./bin_means/gmm_wide.npz')['t_means_w']

    if GMM_ANSATZ:

        # fit the ensemble of intervals
        if READ_INTERVALS:  # this is the original ensemble that created results in paper
            intervals_full_rank_gmm_ans_files = np.load(
                file='./data/wide_bin_deconvolution/intervals_osb_po_full_rank_misspec_gmm_ansatz.npz'
            )
            intervals_osb_fr = intervals_full_rank_gmm_ans_files['intervals_osb_fr']
            intervals_po_fr = intervals_full_rank_gmm_ans_files['intervals_po_fr']
        else:
            pass

        # estimate the coverage
        coverage_osb_fr = compute_coverage(
            intervals=intervals_osb_fr,
            true_bin_means=t_means_w
        )
        coverage_po_fr = compute_coverage(
            intervals=intervals_po_fr,
            true_bin_means=t_means_w
        )

        # save the above
        np.savez(
            file='./data/wide_bin_deconvolution/coverage_osb_po_full_rank_misspec_gmm_ansatz',
            coverage_osb_fr=coverage_osb_fr,
            coverage_po_fr=coverage_po_fr
        )