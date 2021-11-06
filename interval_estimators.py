"""
Computational implementation of the following intervals:
1. Least-Squares
2. OSB
3. PO
4. SSB
5. Fixed-Width Minimax bounds

Author        : Michael Stanley
Created       : 05 Nov 2021
Last Modified : 06 Nov 2021
===============================================================================
"""
import numpy as np
from scipy import stats

def ls_point_estimator(K, y):
    """
    True histogram bin count point estimator when K has full rank.

    Parameters:
    -----------
        K (np arr) : m x n smearing matrix - n # true bins, n # smear bins
        y (np arr) : m x 1 observation vector

    Returns:
    --------
        least squares estimator

    """
    return np.linalg.inv(K.T @ K) @ K.T @ y


def ls_estimator_cov(K):
    """
    Compute the covariance of the estimator for K full column rank

    NOTE:
    - in practice, real observations should be used instead of
      K @ true_means
    - this function expects that the noise covariance is the identity matrix

    Parameters:
    -----------
        K          (np arr) : m x n smearing matrix - n # true bins,
                              m # smear bins

    Returns:
    --------
        n x n covariance matrix
    """
    KTKinv = np.linalg.inv(K.T @ K)

    return KTKinv


def least_squares_interval(K, h, y, alpha=0.05):
    """
    Creates a 1 - alpha/2 confidence interval for functional of interest.

    NOTE:
    - uses the normal approximation to the poisson distribution
    - this function expects that the noise covariance is the identity matrix

    Parameters:
    -----------
        K          (np arr) : mxn  smearing matrix - n # true bins,
                              n # smear bins
        h          (np arr) : n x 1 functional for original bins
        y          (np arr) : m x 1 observation vector
        alpha      (float)  : type 1 error threshold -- (1 - confidence level)

    Returns:
    --------
        tuple of lower and upper bound of 1 - alpha/2 confidence interval.
    """
    lambda_hat = ls_point_estimator(K=K, y=y)
    lambda_cov = ls_estimator_cov(K=K)
    crit_val = stats.norm.ppf(q=(1 - (alpha / 2)))

    lb = np.dot(h, lambda_hat) - crit_val * np.sqrt(h.T @ lambda_cov @ h)
    ub = np.dot(h, lambda_hat) + crit_val * np.sqrt(h.T @ lambda_cov @ h)

    return (lb, ub)

