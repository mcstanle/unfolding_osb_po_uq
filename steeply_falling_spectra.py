'''
This file is meant to contain the functions to perform the steeply falling
spectra experiments. For details on the experimental setup, see Kuusela 2016.

Author   : Mike Stanley
Created  : 01 June 2021
Modified : 16 Nov 2021
===============================================================================
'''
from functools import partial
import json
import numpy as np
from scipy import stats
from scipy.integrate import quad
from tqdm import tqdm


def f_sf(
    p_T, L, N_0, alpha, sqrt_s, beta, gamma
):
    """ Particle Level intensity function for the inclusive jet spectrum """
    if (p_T > 0) & (p_T <= sqrt_s / 2):
        return L * N_0 * (p_T ** (-alpha)) * (1 - (2 / sqrt_s) * p_T)**beta *\
            np.exp(- gamma / p_T)
    else:
        return 0


def hetsced_noise_eq(p_T, N=1, S=1, C=0.05):
    """ Solves for variance as a function of the inputs (see eq.3.37) """
    return np.square(p_T) * (np.square(N / p_T) + np.square(S / np.sqrt(p_T)) + np.square(C))


def k(p_T_prime, p_T, N=1, S=1, C=0.05):
    """ Compute the kernel between p_T_prime and p_T """
    variance = hetsced_noise_eq(p_T=p_T, N=N, S=S, C=C)
    coeff = (np.sqrt(2 * np.pi * variance)) ** (-1)
    exp_term = np.exp(-(1 / (2 * variance)) * np.square(p_T_prime - p_T))

    return coeff * exp_term


def smear_integrand(p_T, p_T_prime, f_intensity):
    """ integrand of Fredholm integral """
    f_eval = f_intensity(p_T=p_T)
    kernel_eval = k(p_T_prime=p_T_prime, p_T=p_T)

    return kernel_eval * f_eval


def g(
    p_T_prime, lb, ub, f_intensity
):
    """ Compute the smeared intensity at point p_T_prime """
    return quad(
        func=smear_integrand,
        args=(p_T_prime, f_intensity),
        a=lb, b=ub
    )[0]


def g_over_interval(
    smear_grid,
    true_lb,
    true_ub,
    f_intensity
):
    """ Numerically solves g integral over interval for p_T_prime """
    g_evals = np.zeros_like(smear_grid)

    for i, int_grid_i in enumerate(smear_grid):
        g_evals[i] = g(
            p_T_prime=int_grid_i,
            lb=true_lb, ub=true_ub,
            f_intensity=f_intensity
        )

    return g_evals


def compute_means(f_intensity, true_grid, smear_grid):
    """ Compute means in both true and smeared space """
    m = true_grid.shape[0] - 1
    n = smear_grid.shape[0] - 1

    true_means = np.zeros(m)
    smear_means = np.zeros(n)

    for i in range(m):
        true_means[i] = quad(
            func=f_intensity,
            a=true_grid[i],
            b=true_grid[i + 1]
        )[0]

    for i in range(n):
        smear_means[i] = quad(
            func=g,
            args=(true_grid[0], true_grid[-1], f_intensity),
            a=smear_grid[i],
            b=smear_grid[i + 1]
        )[0]

    return true_means, smear_means


def inner_int(p_T, F_i_lower, F_i_upper):
    """ Compute the inner integral for the K_{ij} approximation """
    std = np.sqrt(hetsced_noise_eq(p_T))
    return stats.norm.cdf((F_i_upper - p_T) / std) - stats.norm.cdf((F_i_lower - p_T) / std)


def compute_K(f_intensity, s_edges, t_edges):
    """
    Use full computation and no piece-wise approx
    
    Indices:
    - i indexes rows of K, i.e. smear bins
    - j indexes columns of K, i.e. true bins
    """
    dim_smear = s_edges.shape[0] - 1
    dim_true = t_edges.shape[0] - 1
    K = np.zeros(shape=(dim_smear, dim_true))

    for j in tqdm(range(dim_true)):

        # compute the denominator
        denom_eval = quad(
            f_intensity,
            a=t_edges[j],
            b=t_edges[j + 1]
        )

        for i in range(dim_smear):
            int_eval = quad(
                func=lambda x: f_intensity(x) * inner_int(
                    x,
                    F_i_lower=s_edges[i],
                    F_i_upper=s_edges[i + 1]
                ),
                a=t_edges[j],
                b=t_edges[j + 1]
            )

            K[i, j] = int_eval[0] / denom_eval[0]

    return K


def gen_poisson_data(mu):
    """ create one realization of data given vector of bin means """
    data = np.zeros_like(mu)
    for i in range(mu.shape[0]):
        data[i] = stats.poisson(mu=mu[i]).rvs()

    return data


def compute_intensities():
    """
    Returns the predefined intensity functions from 
    './simulation_model_parameters.json'.
    """
    with open('./simulation_model_parameters.json') as f:
        parameters = json.load(f)

    true_params = parameters['steep_f_spec_truth']
    mc_params = parameters['steep_f_spec_ansatz']

    f_true = partial(
        f_sf, 
        L=true_params['L'],
        N_0=true_params['N_0'],
        alpha=true_params['ALPHA'],
        sqrt_s=true_params['SQRT_S'],
        beta=true_params['BETA'],
        gamma=true_params['GAMMA']
    )
    f_ans = partial(
        f_sf, 
        L=mc_params['L'],
        N_0=mc_params['N_0'],
        alpha=mc_params['ALPHA'],
        sqrt_s=mc_params['SQRT_S'],
        beta=mc_params['BETA'],
        gamma=mc_params['GAMMA']
    )

    return f_true, f_ans