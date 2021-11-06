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
import cvxpy as cp
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


def osb_interval(
    y, K, h, A, alpha=0.05, verbose=False, options={}
):
    """
    Compute OSB interval.

    Dimension key:
    - n : number of smear bins
    - m : number of true bins

    Parameters:
    -----------
        y         (np arr) : cholesky transformed data -- n x 1
        K         (np arr) : cholesky transformed matrix -- n x m
        h         (np arr) : functional for parameter transform -- m x 1
        A         (np arr) : Matrix to enforce non-trivial constraints
        alpha     (float)  : type 1 error threshold -- (1 - confidence level)
        options   (dict)   : ECOS options for cvxpy

    Returns:
    --------
        opt_lb, -opt_ub (tup) : lower and upper interval bounds
    """
    m = K.shape[1]

    # find the slack factor
    x = cp.Variable(m)
    cost = cp.sum_squares(y - K @ x)
    prob = cp.Problem(
        objective=cp.Minimize(cost),
        constraints=[
            A @ x <= 0
        ]
    )
    s2 = prob.solve(solver='ECOS', verbose=verbose, **options)

    # find the constraint bound
    sq_err_constr = np.square(
        stats.norm(loc=0, scale=1).ppf(1 - (alpha / 2))
    ) + s2

    # define a variables to solve the problem
    x_lb = cp.Variable(m)
    x_ub = cp.Variable(m)

    # define the problem
    prob_lb = cp.Problem(
        objective=cp.Minimize(h.T @ x_lb),
        constraints=[
            cp.square(cp.norm2(y - K @ x_lb)) <= sq_err_constr,
            A @ x_lb <= 0
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(-h.T @ x_ub),
        constraints=[
            cp.square(cp.norm2(y - K @ x_ub)) <= sq_err_constr,
            A @ x_ub <= 0
        ]
    )

    # solve the problem
    opt_lb = prob_lb.solve(solver='ECOS', verbose=verbose, **options)
    opt_ub = prob_ub.solve(solver='ECOS', verbose=verbose, **options)

    # check for convergence
    assert 'optimal' in prob_lb.status
    assert 'optimal' in prob_ub.status

    return opt_lb, -opt_ub


def prior_interval_construct(y, opt_x_lb, opt_x_ub, b, alpha, m):
    """
    Given the atomic pieces from the optimization, this function constructs
    the actual interval.

    Parameters:
    -----------
        y        (np arr) : observed bin counts - m elements
        opt_x_lb (np arr) : optimized lb x vector - first m elements are w,
                            next n elements are c
        opt_x_ub (np arr) : optimized ub x vector - first m elements are w,
                            next n elements are c
        b        (np arr) : n length vector -- part of physical constraints
        alpha    (float)  : in [0, 1], confidence level of interval
        m        (int)    : number of smear bins

    Return:
    -------
        tuple of optimized lower and upper endpoints
    """
    w_lb = opt_x_lb[:m].copy()
    c_lb = opt_x_lb[m:].copy()
    w_ub = opt_x_ub[:m].copy()
    c_ub = opt_x_ub[m:].copy()

    # compute Gaussian quantile
    z_alpha = stats.norm.ppf(1 - alpha/2)

    # compute bounds
    lb = np.dot(w_lb, y) - z_alpha * np.linalg.norm(w_lb) - np.dot(b, c_lb)
    ub = np.dot(w_ub, y) + z_alpha * np.linalg.norm(w_ub) + np.dot(b, c_ub)

    return lb, ub


def po_interval(
    y,
    prior_mean,
    K,
    h,
    A,
    alpha=0.05,
    return_int=True,
    verbose=False,
    return_cvx_obj=False,
    options={'max_iters': 100, 'abstol': 1e-8, 'reltol': 1e-8, 'feastol': 1e-8}
):
    """
    Prior interval optimization.

    Dimension key
    - m : smear space dimension
    - n : unfolded space dimension
    - q : number of inequality constraints

    NOTE: the optimized variables, x_lb and x_ub, are oriented such that the
    first m elements correspond to w and the next q elements correspond to c.

    Options for the ECOS algo: https://www.cvxpy.org/tutorial/advanced/index.html

    Parameters:
    -----------
        y              (np arr) : m x 1 
        prior_mean     (np arr) : n x 1
        K              (np arr) : smearing matrix -- m x n
        h              (np arr) : functional -- n x 1
        A              (np arr) : constraint matrix -- q x n
        alpha          (float)  : confidence level
        return_int     (bool)   : toggle to return interval or optimized variables
        verbose        (bool)   : returns the output of the optimization
        return_cvx_obj (bool)   : return the cvxpy objects
        options        (dict)   : options for the ECOS algo (see above for link)

    Returns:
    --------
        if return_int:
            tuple : (lower bound, upper bound)
        else:
            optimized vectors x_lb.value, x_ub.value
    """
    # define the new algebra objects
    m, n = K.shape
    q = A.shape[0]

    P = np.zeros(shape=(m + q, m + n))
    gamma = np.zeros(m + n)
    B = np.zeros(shape=(m + q, m + q))
    F = np.zeros(shape=(q, m + q))
    D_lb = np.zeros(shape=(n, m + q))
    D_ub = np.zeros(shape=(n, m + q))

    P[:m, :n] = K.copy()

    D_lb[:, :m] = - K.copy().T
    D_lb[:, m:] = A.copy().T
    D_ub[:, :m] = - K.copy().T
    D_ub[:, m:] = - A.copy().T

    gamma[:n] = prior_mean.copy()
    B[:m, :m] = np.identity(m)
    F[:, m:] = np.identity(q)

    # compute the normal quantile
    norm_quant = stats.norm.ppf(1 - alpha/2)

    # set up optimizations
    x_lb = cp.Variable(m + q)
    x_ub = cp.Variable(m + q)

    prob_lb = cp.Problem(
        objective=cp.Maximize(x_lb.T @ P @ gamma - norm_quant * cp.norm(B.T @ x_lb)),
        constraints=[
            h + D_lb @ x_lb == np.zeros(n),
            F @ x_lb >= np.zeros(q)
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(x_ub.T @ P @ gamma + norm_quant * cp.norm(B.T @ x_ub)),
        constraints=[
            h + D_ub @ x_ub == np.zeros(n),
            F @ x_ub >= np.zeros(q)
        ]
    )

    # solve the problems
    opt_lb = prob_lb.solve(verbose=verbose, **options)
    opt_ub = prob_ub.solve(verbose=verbose, **options)

    # check convergence
    assert 'optimal' in prob_lb.status
    assert 'optimal' in prob_ub.status

    if return_int:
        return prior_interval_construct(
            y=y,
            opt_x_lb=x_lb.value,
            opt_x_ub=x_ub.value,
            b=np.zeros(q),
            alpha=alpha,
            m=m
        )
    elif return_cvx_obj:
        return prob_lb, prob_ub
    else:
        return x_lb.value, x_ub.value


def ssb_interval(y, K, h, A, alpha, solver='ECOS', verbose=False, options={}):
    """
    SSB interval optimization.

    NOTE:
    - data and K matrix are assumed to be Cholesky transformed

    Dimension key:
        m : number of smear bins
        n : number of unfold bins
        q : number of constraints

    Parameters:
    -----------
        y            (np arr) : m element array Cholesky trans observations
        K            (np arr) : mxn smearing matrix
        h            (np arr) : n element functional on the parameters
        A            (np arr) : q x n constraint matrix
        alpha        (float)  : interval level
        solver       (str)    : optimizer method for cvxpy
        options      (dict)   : ECOS options for cvxpy

    Returns:
    --------
        tuple -- lower/upper bound
    """
    # dimensions of problem
    m, n = K.shape
    
    # find the chi-sq critical value
    chisq_q = stats.chi2(df=m).ppf(1 - alpha)
    
    # declare the variables to be optimized
    x_lb = cp.Variable(n)
    x_ub = cp.Variable(n)
    
    # define the constraints
    constraints_lb = [
        cp.norm2(y - K @ x_lb) ** 2 <= chisq_q,
        A @ x_lb <= 0
    ]
    constraints_ub = [
        cp.norm2(y - K @ x_ub) ** 2 <= chisq_q,
        A @ x_ub <= 0
    ]
    
    # set up the optimization problems
    prob_lb = cp.Problem(
        objective=cp.Minimize(h.T @ x_lb),
        constraints=constraints_lb
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(-h.T @ x_ub),
        constraints=constraints_ub
    )
    
    # solve the problem
    opt_lb = prob_lb.solve(solver=solver, verbose=verbose, **options)
    opt_ub = prob_ub.solve(solver=solver, verbose=verbose, **options)
    
    # determine if optimal solution is found
    assert 'optimal' in prob_lb.status
    assert 'optimal' in prob_ub.status

    return opt_lb, -opt_ub


def minimax_interval_radius_bounds(h, K, alpha):
    """
    A lowerbound/upperbound on the half-length of the Donoho minimax intervals computed
    with cvxpy.

    NOTE:
    - K matrix are assumed to be Cholesky transformed
    - SINCE it is cholesky transformed, we are assuming that the value of sigma
      is 1.

    Parameters:
    -----------
        K            (np arr) : mxn smearing matrix - cholesky transformed
        h            (np arr) : n element functional on the parameters
        alpha        (float)  : interval level

    Returns:
    --------
        tuple -- lower/upper bound, lower/upper bounds optimized variables
    """
    # find relevant gaussian quantile
    g_quant_lb = stats.norm.ppf(1 - alpha)
    g_quant_ub = stats.norm.ppf(1 - alpha / 2)

    # create the augmented K matrix
    pad_mat = np.concatenate(
        (
            np.eye(N=K.shape[1], M=K.shape[1]),
            -np.eye(N=K.shape[1], M=K.shape[1])
        ), axis=1
    )
    K_aug = K @ pad_mat

    # create the augmented functional
    L_func = np.concatenate((h, -h))
    z_dim = K.shape[1] * 2

    # lower bound
    z_lb = cp.Variable(z_dim)
    moc_prob_lb = cp.Problem(
        objective=cp.Maximize(L_func @ z_lb),
        constraints=[
            cp.norm(K_aug @ z_lb) <= 2 * g_quant_lb,
            z_lb >= np.zeros(z_dim),
        ]
    )
    moc_prob_lb.solve()

    # upper bound
    z_ub = cp.Variable(z_dim)
    moc_prob_ub = cp.Problem(
        objective=cp.Maximize(L_func @ z_ub),
        constraints=[
            cp.norm(K_aug @ z_ub) <= 2 * g_quant_ub,
            z_ub >= np.zeros(z_dim),
        ]
    )
    moc_prob_ub.solve()

    # determine if optimal solution is found
    assert moc_prob_lb.status == 'optimal'
    assert moc_prob_ub.status == 'optimal'

    return moc_prob_lb.value, moc_prob_ub.value, z_lb, z_ub