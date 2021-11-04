"""
This script is meant to parallelize the process of generating new data in order
to find a Least squares generated ansatz function that has optimally bad
coverage.

TODO: update the names of the referenced functions and such.

Author        : Mike Stanley
Created       : 20 August 2021
Last Modified : 20 August 2021
===============================================================================
"""
import cvxpy as cp
import numpy as np
import multiprocessing as mp
from scipy import interpolate
from simulate_coverage import least_squares_interval, int_covers_truth
from time import time
from tqdm import tqdm
from unfolding_utils import compute_K_arbitrary, generate_hists


def read_data(data_path):
    """
    Reads in relevant data for the coverage computation.

    Includes the follwing
    1. Coverage data (rows are for simulations, columns are smear bin counts)
    2. True K matrix
    3. True bin means
    4. Smear bin means --- NOTE: this is new relative to the old version
    5. unfold and smear bin edges
    6. data generating parameters
    7. bin functionals

    NOTE: These data were prepared in
    ~/Research/strict_bounds/prior_optimized_paper/brute_force_ansatz_GMM_only.ipynb

    Parameters:
    -----------
        data_path (string) : path to data

    Returns:
    --------
        see below....
    """
    data_package = np.load(
        file=data_path
    )

    return (
        data_package['coverage_data'],
        data_package['K'],
        data_package['true_bin_means'],
        data_package['smear_bin_means'],
        data_package['unfold_edges'],
        data_package['smear_edges'],
        data_package['pi'],
        data_package['mu'],
        data_package['sigma'],
        data_package['T'],
        data_package['sigma_smear'],
        data_package['H']
    )


def generate_ansatz_data(
    random_seed,
    mu, sigma, T, sigma_smear, bin_lb, bin_ub, num_bins=(40, 40)
):
    """
    Draw from the distribution that will create the ansatz intensity function.

    Parameters:
    -----------
        random_seed (int)    : random seed to control data generation
        mu          (np arr) : 2x1 GMM means
        sigma       (np arr) : 2x1 GMM standard deviations
        T           (int)    : total mean number of events
        sigma_smear (float)  : smearing kernel strength
        bin_lb      (float)  : lower bound of bins
        bin_ub      (float)  : upper bound of bins
        num_bins    (tuple)  : number of real and smeared bins, respectively
    Returns:
    --------
    """
    np.random.seed(random_seed)
    hist_true, hist_smear = generate_hists(
        pi=pi,
        mu=mu,
        sigma=sigma,
        T=T,
        sigma_smear=sigma_smear,
        bin_lb=bin_lb,
        bin_ub=bin_ub,
        num_bins_real=[num_bins[0]],
        num_bins_smear=[num_bins[1]]
    )

    return hist_smear[0][0]


def constrained_ls_estimator(data, K, smear_means, dim=40):
    """
    Fits least squared solution with positivity constraint on the
    parameters;

    Parameters:
    -----------
        data        (np arr) : sampled data to create ansatz
        K           (np arr) : smearing matrix
        smear_means (np arr) : smear bin means for computing chol transform
        dim         (int)    : dimension of the parameters

    Returns:
    --------
        x_opt.value (np arr) : constrained least squares estimate
    """
    # compute cholesky transformed K matrix from data
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)
    K_tilde = L_data_inv @ K

    # transform the data
    y = L_data_inv @ data

    # perform the optimization
    x_opt = cp.Variable(dim)
    ls_constr_prob = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - K_tilde @ x_opt)),
        constraints=[
            x_opt >= 0
        ]
    )
    x_opt_sol = ls_constr_prob.solve()

    # check convergence
    ls_constr_prob.status == 'optimal'

    return x_opt.value


def fit_interpolator_intensity(unfold_edges, ls_est_vals):
    """
    Fit the intensity function based on an intepolation of the least squares
    estimator.

    Interpolates with a cubic spline from scipy.

    NOTE: this version ensures that the intensity function is always positive.

    Parameters:
    -----------
        unfold_edges (np arr) : edges of unfolding bins
        ls_est_vals  (np arr) : constrained least squares estimate

    Returns:
    --------
        interp_ansatz_intensity (func) : float input function of intensity function
    """
    # find the width for intensity scaling
    width = unfold_edges[1] - unfold_edges[0]

    # interpolate the above
    interp_x_vals = (unfold_edges[:-1] + unfold_edges[1:])/2
    interp_ansatz = interpolate.CubicSpline(
        x=interp_x_vals,
        y=ls_est_vals
    )

    interp_ansatz_intensity = lambda x: np.max([interp_ansatz(x) / width, 0])

    return interp_ansatz_intensity


def compute_new_K(
    interp_intensity, dims, smear_edges, unfold_edges, sigma_smear
):
    """
    Fit the new K matrix with the interpolator intensity function.

    Parameters:
    -----------
        interp_intensity (func)   : intensity function
        dims             (tuple)  : smear and unfold dimensions
        smear_edges      (np arr) : edges of smeared bins
        unfold_edges     (np arr) : edges of true bins
        sigma_smear      (float)  : smearing strength

    Returns:
    --------
        computed K matrix (np arr)
    """
    return compute_K_arbitrary(
        intensity_func=interp_intensity,
        dim_smear=dims[0],
        dim_unfold=dims[1],
        s_edges=smear_edges,
        u_edges=unfold_edges,
        sigma_smear=sigma_smear
    )


def compute_coverage(K, H, data_arr, true_means, alpha):
    """
    For a given ansatz matrix compute the coverage by bin
    
    Parameters
    ----------
    K          (np arr) : smearing matrix
    H          (np arr) : bin functionals
    data_arr   (np arr) : NON-transformed data
    true_means (np arr) : true process means
    alpha      (float)  : level of interval
    
    Returns
    -------
    coverage (np arr) : one values for each bin
    """
    NUM_BINS = H.shape[0]
    M = data_arr.shape[0]
    
    # compute the true wide bin means
    wide_bin_means = H @ true_means
    
    # find the least squares intervals and coverage
    coverage_ls = np.zeros(shape=(M, NUM_BINS))

    # compute the cholesky transform
    Sigma = np.diag(K @ true_means)
    L = np.linalg.cholesky(Sigma)
    L_inv = np.linalg.inv(L)

    for i in range(M):  # simulation number
        
        # cholesky transform the data and matrix
        y_i = L_inv @ data_arr[i]
        K_i = L_inv @ K
        
        for j in range(NUM_BINS):    # functional bin number

            # compute the coverage
            interval_ij = least_squares_interval(
                K=K_i, h=H[j, :],
                true_means=true_means,
                y=y_i, alpha=alpha
            )
            
            # coverage
            coverage_ls[i, j] += int_covers_truth(wide_bin_means[j], interval_ij)
            
    # compute coverage for each bin
    coverage = coverage_ls.mean(axis=0)
    
    return coverage


def run_pipeline(
    i, random_seed,
    mu, sigma, T, sigma_smear, bin_lb, bin_ub,
    K,
    unfold_edges, smear_edges,
    H, data_cov, true_means,
    alpha
):
    """
    Run full pipeline of sampling data, generate ansatz, and estimating coverage.

    Parameters:
    -----------
        i            (int)    : index for iteration
        random_seed  (int)    : random seed for generating the data to create ansatz
        mu           (np arr) : GMM means
        sigma        (np arr) : GMM standard deviations
        T            (int)    : expected number of total events
        sigma_smear  (float)  : smearing strength
        bin_lb       (float)  : lower bound of range 
        bin_ub       (float)  : upper bound of range
        K            (np arr) : true smearing matrix
        unfold_edges (np arr) : edges of the unfolded bins
        smear_edges  (np arr) : edges of the smeared bins
        H            (np arr) : functionals
        data_cov     (np arr) : data for estimating coverage
        true_means   (np arr) : true bin means
        alpha        (float)  : significance level of intervals

    Returns:
    --------
        data_i   (np arr) : data used to generate the ansatz
        coverage (np arr) : estimated coverage
    """
    # generate data
    data_i = generate_ansatz_data(
        random_seed=random_seed,
        mu=mu, sigma=sigma, T=T, sigma_smear=sigma_smear,
        bin_lb=bin_lb, bin_ub=bin_ub
    )

    # find constrained LS estimator
    x_opt = constrained_ls_estimator(
        data=data_i,
        K=K,
        smear_means=K @ true_means
    )

    # create interpolator intensity function
    interp_ansatz_intensity = fit_interpolator_intensity(
        unfold_edges=unfold_edges, ls_est_vals=x_opt
    )

    # compute new K matrix
    K_ansatz = compute_new_K(
        interp_intensity=interp_ansatz_intensity,
        dims=(len(smear_edges) - 1, len(unfold_edges) - 1),
        smear_edges=smear_edges,
        unfold_edges=unfold_edges,
        sigma_smear=sigma_smear
    )

    # estimate binwise coverage
    coverage = compute_coverage(
        K=K_ansatz,
        H=H,
        data_arr=data_cov,
        true_means=true_means,
        alpha=alpha
    )

    return i, data_i, coverage


def parallelize_pipe(
    random_seeds,
    mu, sigma, T, sigma_smear, bin_lb, bin_ub,
    K,
    unfold_edges, smear_edges,
    H, data_cov, true_means,
    alpha
):
    """
    Parallelize the run_pipeline() function

    Parameters:
    -----------
        random_seeds (list) : list of integer random seeds
        mu           (np arr) : GMM means
        sigma        (np arr) : GMM standard deviations
        T            (int)    : expected number of total events
        sigma_smear  (float)  : smearing strength
        bin_lb       (float)  : lower bound of range 
        bin_ub       (float)  : upper bound of range
        K            (np arr) : true smearing matrix
        unfold_edges (np arr) : edges of the unfolded bins
        smear_edges  (np arr) : edges of the smeared bins
        H            (np arr) : functionals
        data_cov     (np arr) : data for estimating coverage
        true_means   (np arr) : true bin means
        alpha        (float)  : significance level of intervals

    Returns:
    --------
        generated ansatz data
        coverage vectors
    """
    pool = mp.Pool(mp.cpu_count())

    print('Number of available CPUs: %i' % mp.cpu_count())

    # storage for run_pipeline() output
    output_data = []

    def collect_data(data):
        output_data.append(data)

    for i, rs_i in enumerate(random_seeds):
        pool.apply_async(
            run_pipeline,
            args=(
                i,
                rs_i,
                mu, sigma, T, sigma_smear, bin_lb, bin_ub,
                K, unfold_edges, smear_edges,
                H,
                data_cov, true_means,
                alpha
            ),
            callback=collect_data
        )

    pool.close()
    pool.join()

    output_data.sort(key=lambda x: x[0])
    ansatz_data_lst = [_[1] for _ in output_data]
    coverages_lst = [_[2] for _ in output_data]

    return np.array(ansatz_data_lst), np.array(coverages_lst)


if __name__ == "__main__":

    # read in the data
    DATA_PATH = './data/brute_force_ansatz_data_package_GMM_ONLY.npz'
    coverage_data, K, true_bin_means, smear_bin_means, unfold_edges, smear_edges, pi, mu, sigma, T, sigma_smear, H = read_data(DATA_PATH)
    print('Coverage Data shape: %s' % str(coverage_data.shape))

    # operational parameters
    NUM_TRIES = 1000  # number of brute force attempts
    ALPHA = 0.05

    # create random seeds
    random_seeds = np.arange(NUM_TRIES)

    # compute coverages
    START = time()
    ansatz_data, coverages = parallelize_pipe(
        random_seeds=random_seeds,
        mu=mu, sigma=sigma, T=T, sigma_smear=sigma_smear,
        bin_lb=unfold_edges[0], bin_ub=unfold_edges[-1],
        K=K,
        unfold_edges=unfold_edges, smear_edges=smear_edges,
        H=H, data_cov=coverage_data, true_means=true_bin_means,
        alpha=ALPHA
    )
    END = time()

    print('Time to generate %i coverage vectors: %.4f mins' % (
        NUM_TRIES, ((END - START) / 60)
    ))

    # save the above
    np.save(file='./data/ansatz_data_gmm_only.npy', arr=ansatz_data)
    np.save(file='./data/coverages_gmm_only.npy', arr=coverages)