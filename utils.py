"""
Utility functions for code in this directory.

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 05 Nov 2021
===============================================================================
"""
import numpy as np
from scipy import stats
from scipy.integrate import quad


def compute_GMM_bin_means(
    true_edges,
    intensity_func,
    pi,
    mu,
    sigma,
    T,
    K
):
    """
    With some GMM intensity function and some domain bidding, compute the mean
    count for each bin in both the true and smeared spaces.

    Parameters:
        true_edges     (np arr)   : edges of bins for true histogram
        intensity_func (function) : intensity function for the poisson point
                                    process
        pi             (np arr)   : mixing probabilties for each gaussian
        mu             (np arr)   : mean for each gaussian component
        sigma          (np arr)   : standard deviation for each component
        T              (np arr)   : Mean of poisson process
        K              (np arr)   : smearing matrix

    Returns:
        bin mean counts
        - true_means  (np arr)
        - smear_means (np arr)
    """
    NUM_REAL_BINS = true_edges.shape[0] - 1
    true_means = np.zeros(NUM_REAL_BINS)

    for i in range(NUM_REAL_BINS):
        true_means[i] = quad(
            func=intensity_func,
            args=(pi, mu, sigma, T),
            a=true_edges[i],
            b=true_edges[i + 1]
        )[0]

    smear_means = K @ true_means

    return true_means, smear_means

def compute_even_space_bin_edges(bin_lb, bin_ub, num_bins):
    """
    Computes the bin endpoints for evenly spaced bins.

    Parameters:
    -----------
        bin_lb   (float) : lower bound on bins
        bin_ub   (float) : upper bound on bins
        num_bins (int)   : number of bins

    Returns:
    --------
        bin edges (np arr) : has length (num_bins + 1) since includes both the
                             endpoints.
    """
    return np.linspace(bin_lb, bin_ub, num=num_bins + 1)

def compute_K_gmm(
    intensity_func, dim_smear, dim_true, s_edges, t_edges, sigma_smear,
    pi, mu, sigma, T
):
    """
    Compute the smearing matrix K.

    NOTE: assumes that intensity function takes the pi, mu, sigma, and T args

    Parameters:
        intensity_func (function) : Poisson process intensity function
        dim_smear      (int)      : number of bins in the smeared space
        dim_true       (int)      : number of bins in the true space
        s_edges        (np array) : smeared space bin edges
        t_edges        (np array) : true space bin edges
        sigma_smear    (float)    : convolution standard deviation
        pi             (np arr)   : GMM mixing weights
        mu             (np arr)   : GMM means
        sigma          (np arr)   : GMM standard devs
        T              (int)      : Poisson proc mean over whole space

    Returns:
        K (np array) : dimension dim_smear X dim_true
    """
    K = np.zeros(shape=(dim_smear, dim_true))

    for j in range(dim_true):

        # compute the denominator
        denom_eval = quad(
            func=intensity_func,
            a=t_edges[j],
            b=t_edges[j + 1],
            args=(pi, mu, sigma, T)
        )

        for i in range(dim_smear):

            # compute the numerator
            int_eval = quad(
                func=lambda x: intensity_func(
                    x, pi=pi, mu=mu, sigma=sigma, T=T
                ) * inner_int(
                    x, S_lower=s_edges[i], S_upper=s_edges[i + 1],
                    sigma=sigma_smear
                ),
                a=t_edges[j],
                b=t_edges[j + 1]
            )

            K[i, j] = int_eval[0] / denom_eval[0]

    return K


def compute_K_arbitrary(
    intensity_func,
    s_edges,
    t_edges,
    sigma_smear
):
    """
    Compute the K matrix using an arbitrary intensity function.

    NOTE: assumes that intensity function is computationally defined and is
    thus simply evaluated at each x.

    Parameters:
        intensity_func (function) : intensity function
        s_edges        (np array) : smeared space bin edges
        t_edges        (np array) : true space bin edges
        sigma_smear    (float)    : convolution standard deviation

    Returns:
        K (np array) : dimension dim_smear X dim_unfold
    """
    # find the true and smear dimensions
    dim_true = t_edges.shape[0] - 1
    dim_smear = s_edges.shape[0] - 1

    # compute the smearing matrix
    K = np.zeros(shape=(dim_smear, dim_true))

    for j in range(dim_true):

        # compute the denominator
        denom_eval = quad(
            func=intensity_func,
            a=t_edges[j],
            b=t_edges[j + 1]
        )

        for i in range(dim_smear):

            # compute the numerator
            int_eval = quad(
                func=lambda x: intensity_func(x) * inner_int(
                    x, S_lower=s_edges[i], S_upper=s_edges[i + 1],
                    sigma=sigma_smear
                ),
                a=t_edges[j],
                b=t_edges[j + 1]
            )

            K[i, j] = int_eval[0] / denom_eval[0]

    return K


def inner_int(y, S_lower, S_upper, sigma):
    """
    Find the inner integral of the K matrix component calc

    Parameters:
        y       (float) : variable of which we want the inner integral to
                          sbe a function
        S_lower (float) : lower bound of segment
        S_upper (float) : upper bound of segment
        sigma   (float) : convolution standard deviation

    Returns:
        float of inner integral evaluated at y
    """
    return stats.norm.cdf((S_upper - y)/sigma) - stats.norm.cdf((S_lower - y)/sigma)

def intensity_f(x, pi, mu, sigma, T):
    """
    Evaluate the intensity function at x.

    In accordance with the data generating process, the intensity function is
    based on the GMM.

    Parameters:
        x     (float)  : value at which to evaluate the intensity func
        pi    (np arr) : mixture components for the GMM
        mu    (np arr) : Means for gaussian components
        sigma (np arr) : Standard deviations for the gaussian components
        T     (int)    : number of poisson data realizations

    Returns:
        float of intensity function evaluation
    """
    norm0 = stats.norm(loc=mu[0], scale=sigma[0])
    norm1 = stats.norm(loc=mu[1], scale=sigma[1])

    return pi[0] * T * norm0.pdf(x) + pi[1] * T * norm1.pdf(x)


def int_covers_truth(truth, interval):
    """
    Determine if a given interval covers the true functional value.

    Parameters:
    -----------
        truth    (float) : the actual functional value
        interval (tuple) : optimized interval for the functional

    Returns:
    --------
        1 if the interval covers the truth
    """
    covers = True
    if truth < interval[0]:
        covers = False
    if truth > interval[1]:
        covers = False

    return int(covers)


def compute_coverage(intervals, true_bin_means):
    """
    Given an ensemble of bin-wise intervals and the true bin count values,
    estimate bin-wise coverage.

    Dimension Key:
        N - number of ensemble elements
        M - number of functionals

    Parameters:
    -----------
        intervals      (np arr) : N x M x 2
        true_bin_means (np arr) : expected counts of true bins

    Returns:
    --------
        coverage (np arr)
    """
    num_sims = intervals.shape[0]
    num_funcs = intervals.shape[1]
    
    # find coverage for each bin
    coverage = np.zeros(10)

    for j in range(num_funcs):
        num_cover_j = 0
        for i in range(num_sims):
            num_cover_j += int_covers_truth(true_bin_means[j], intervals[i, j, :])

        coverage[j] = num_cover_j / num_sims

    return coverage