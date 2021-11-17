"""
Estimate interval coverage for OSB/PO/SSB intervals across shape constraints
and functionals.

When run, the estimated coverages can be found in
./data/steeply_falling_spectrum/estimated_coverages.npz

Author        : Michael Stanley
Created       : 17 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
import numpy as np
from utils import int_covers_truth


def coverage_detector(truth, interval):
    """ wrapper for int_covers_truth, but takes [0, 0] bad intervals into account """
    if (interval[0] == 0.0) & (interval[1] == 0.0):
        return -1
    else:
        return int_covers_truth(truth, interval)


def compute_coverage(coverage_arr):
    """ computes coverage across all bins """
    num_bins = coverage_arr.shape[1]
    coverage_output = np.zeros(num_bins)
    
    for j in range(num_bins):
        x = coverage_arr[:, j]
        coverage_output[j] = x[x != -1].mean()
        
    return coverage_output


def compute_coverage_dict(interval_dict, true_func_vals, num_sims=1000, num_bins=10):
    """
    Estimates coverage from a dictionary of intervals. See the README for an
    explanation regarding the indexing.

    Parameters:
    -----------
        interval_dict  (dict)   : dictionary of intervals
        true_func_vals (np arr) : true bin functional values
        num_sims       (int)    : number of simulations
        num_bins       (int)    : number of functional bins

    Returns:
    --------
        9 numpy arrays, giving functional coverage across intervals/constraints
    """
    # find the coverage for each interval
    coverage_arr_osb_n = np.zeros(shape=(num_sims, num_bins))
    coverage_arr_osb_nd = np.zeros(shape=(num_sims, num_bins))
    coverage_arr_osb_ndc = np.zeros(shape=(num_sims, num_bins))

    coverage_arr_po_n = np.zeros(shape=(num_sims, num_bins))
    coverage_arr_po_nd = np.zeros(shape=(num_sims, num_bins))
    coverage_arr_po_ndc = np.zeros(shape=(num_sims, num_bins))

    coverage_arr_ssb_n = np.zeros(shape=(num_sims, num_bins))
    coverage_arr_ssb_nd = np.zeros(shape=(num_sims, num_bins))
    coverage_arr_ssb_ndc = np.zeros(shape=(num_sims, num_bins))

    for i in range(num_sims):
        for j in range(num_bins):
            
            # osb
            coverage_arr_osb_n[i, j] = coverage_detector(true_func_vals[j], interval_dict['osb|n'][j, i, :])
            coverage_arr_osb_nd[i, j] = coverage_detector(true_func_vals[j], interval_dict['osb|nd'][j, i, :])
            coverage_arr_osb_ndc[i, j] = coverage_detector(true_func_vals[j], interval_dict['osb|ndc'][j, i, :])
            
            # po
            coverage_arr_po_n[i, j] = coverage_detector(true_func_vals[j], interval_dict['po|n'][j, i, :])
            coverage_arr_po_nd[i, j] = coverage_detector(true_func_vals[j], interval_dict['po|nd'][j, i, :])
            coverage_arr_po_ndc[i, j] = coverage_detector(true_func_vals[j], interval_dict['po|ndc'][j, i, :])
            
            # stark
            coverage_arr_ssb_n[i, j] = coverage_detector(true_func_vals[j], interval_dict['ssb|n'][j, i, :])
            coverage_arr_ssb_nd[i, j] = coverage_detector(true_func_vals[j], interval_dict['ssb|nd'][j, i, :])
            coverage_arr_ssb_ndc[i, j] = coverage_detector(true_func_vals[j], interval_dict['ssb|ndc'][j, i, :])

    # compute coverage
    coverage_osb_n = compute_coverage(coverage_arr_osb_n)
    coverage_osb_nd = compute_coverage(coverage_arr_osb_nd)
    coverage_osb_ndc = compute_coverage(coverage_arr_osb_ndc)
    coverage_po_n = compute_coverage(coverage_arr_po_n)
    coverage_po_nd = compute_coverage(coverage_arr_po_nd)
    coverage_po_ndc = compute_coverage(coverage_arr_po_ndc)
    coverage_ssb_n = compute_coverage(coverage_arr_ssb_n)
    coverage_ssb_nd = compute_coverage(coverage_arr_ssb_nd)
    coverage_ssb_ndc = compute_coverage(coverage_arr_ssb_ndc)

    return (
        coverage_osb_n, coverage_osb_nd, coverage_osb_ndc,
        coverage_po_n, coverage_po_nd, coverage_po_ndc,
        coverage_ssb_n, coverage_ssb_nd, coverage_ssb_ndc
    )


if __name__ == "__main__":

    # read in the optimized OSB/PO/SSB intervals
    intervals = np.load(
        file='./data/steeply_falling_spectrum/intervals_optimized_ansatz_rank_def_uneven_bin.npy'
    )

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

    coverage_results = compute_coverage_dict(
        interval_dict=interval_dict,
        true_func_vals=true_func_vals
    )

    # save the above
    np.savez(
        file='./data/steeply_falling_spectrum/estimated_coverages.npz',
        coverage_osb_n=coverage_results[0],
        coverage_osb_nd=coverage_results[1],
        coverage_osb_ndc=coverage_results[2],
        coverage_po_n=coverage_results[3],
        coverage_po_nd=coverage_results[4],
        coverage_po_ndc=coverage_results[5],
        coverage_ssb_n=coverage_results[6],
        coverage_ssb_nd=coverage_results[7],
        coverage_ssb_ndc=coverage_results[8]
    )