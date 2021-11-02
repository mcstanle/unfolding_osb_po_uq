"""
Utility functions for code in this directory.

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 01 Nov 2021
===============================================================================
"""
import numpy as np
from scipy import stats
from scipy.integrate import quad

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