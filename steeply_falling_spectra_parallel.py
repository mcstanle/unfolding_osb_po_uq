"""
Parallelizing the optimization for the following intervals
-- Types
1. OSB
2. PO
3. Stark
-- Constraints
1. Non-negative (n)
2. Non-negative + decreasing (nd)
3. Non-negative + decreasing + convex (ndc)

9 combinations in total

TODO: change the names of the referenced functions and such.

Author        : Mike Stanley
Created       : 31 Aug 2021
Last Modified : 21 Sep 2021
"""
import multiprocessing as mp
import numpy as np
import sys
from time import time
from tqdm import tqdm

from prior_optimized_intervals import prior_interval_opt_cvxpy_A
from simulate_coverage import constrained_interval_cvxpy_A, stark_intervals_cvxpy_A

def interval_opt_i(
    i, index_map, constraint_dict, H, data, K, prior_mean, alpha, verbose, options
):
    """
    Optimizes the correct interval based on the following indexing
    - interval type (3) :
        0 : osb
        1 : po
        2 : stark
    - constraint type (3)
        0 : n
        1 : nd
        2 : ndc
    - functional (10)
        number corresponds to the number of the functional...
    - sample (1000)
        number corresponds to the number of the functional

    NOTE: data and smearing matrix are assumed to be cholesky transformed

    Parameters:
    -----------
        i               (int)    : index to keep track of the intervals
        index_map       (np arr) : defines the mapping from i to interval to optimize
        constraint_dict (dict)   : contains constraint matrices
        H               (np arr) : functional matrix (num functional x unfold dim)
        data            (np arr) : observed data (all simulations -- cholesky trans)
        K               (np arr) : smearing matrix (cholesky trans)
        prior_mean      (np arr) : prior mean for the PO intervals
        alpha           (float)  : prob type 1 error
        verbose         (bool)   : capture output from interval fits
        options         (dict)   : options for the ECOS solver in cvxpy

    Returns:
    --------
        interval (np arr) : two elements - lower bound and upper bound
    """
    # get interval type indices
    int_type, constr_type, functional_num, sample_num = index_map[i].astype(int)

    # get constraint matrix
    A_opt = constraint_dict[constr_type].copy()

    # get functional
    h = H[functional_num, :].copy()

    # get data required
    data_i = data[sample_num, :].copy()

    if verbose:
        print('-----------------------------------')
        print('Interval type: %i | Constr type: %i | func num: %i | sample num: %i' % (
            int_type, constr_type, functional_num, sample_num
            )
        )
        print('Data: %s' % str(data_i))

    if int_type == 0:  # osb
        try:
            opt_interval = constrained_interval_cvxpy_A(
                y=data_i,
                K=K,
                h=h,
                A=A_opt,
                alpha=alpha,
                verbose=verbose,
                options=options
            )
        except:
            opt_interval = (0, 0)
    elif int_type == 1:  # po
        try:
            opt_interval = prior_interval_opt_cvxpy_A(
                y=data_i, prior_mean=prior_mean, K=K,
                h=h, A=A_opt, alpha=alpha, verbose=verbose,
                options=options
            )
        except:
            opt_interval = (0, 0)
    elif int_type == 2:  # stark
        try:
            opt_interval = stark_intervals_cvxpy_A(
                y=data_i, K=K, h=h, A=A_opt, alpha=alpha, verbose=verbose, options=options
            )
        except:
            opt_interval = (0, 0)

    if verbose:
        print('\nOptimized Interval: %s' % str(opt_interval))
    
    return (i, opt_interval)

def parallel_interval_optimize(
    index_map, constraint_dict, H, data, K, prior_mean, alpha, verbose, options
):
    """ Finds time it takes to optimize intervals without parallelization """
    pool = mp.Pool(mp.cpu_count())
    print('Number of cores: %i' % mp.cpu_count())

    intervals = []

    def collect_intervals(interval):
        intervals.append(interval)

    num_idxs = index_map.shape[0]
    print('Number of intervals to fit: %i' % num_idxs)
    print('----------------------------------------')

    for i in range(num_idxs):
        pool.apply_async(
            interval_opt_i,
            args=(
                i,
                index_map,
                constraint_dict,
                H,
                data,
                K,
                prior_mean,
                alpha,
                verbose,
                options
            ),
            callback=collect_intervals
        )

    pool.close()
    pool.join()

    intervals.sort(key=lambda x: x[0])
    intervals_final = [int_i for i, int_i in intervals]

    return np.array(intervals_final)

if __name__ == "__main__":

    # operational constants
    NUM_SIMS = 1000
    ALPHA = 0.05
    CORRECT_K_SPEC = False
    UNFOLD_DIM = 60
    SMEAR_DIM = 30
    NUM_AGG_BINS = 10
    VERBOSE = False

    OPTIMIZER_OPTIONS = {
        'max_iters': 300,
        'abstol': 1e-8,
        'reltol': 1e-8, 
        'feastol': 1e-8
    }

    # starting/ending indices -- use if don't want to fit all intervals
    START_IDX_INT = 0
    START_IDX_CONSTR = 0
    START_IDX_FUNC = 0
    START_IDX_SAMPLE = 0
    END_IDX_INT = 3
    END_IDX_CONSTR = 3
    END_IDX_FUNC = 10
    END_IDX_SAMPLE = NUM_SIMS

    # set base path
    BASE_PATH = '/home/mcstanle/strict_bounds/'

    # read in the relevant data
    files = np.load(
        file=BASE_PATH + '/steeply_falling_spectrum_simulation/data_smear_constr_func_mat_rank_def_uneven_bin.npz'
    )
    data = files['data']
    unfold_means = files['unfold_means']
    smear_means = files['smear_means']
    unfold_means_ansatz = files['unfold_means_ansatz']
    smear_means_ansatz = files['smear_means_ansatz']
    K = files['K']
    K_ansatz = files['K_ansatz']
    A_nn = files['A_nn']
    A_mono = files['A_mono']
    A_con = files['A_con']
    A_nd = files['A_nd']
    A_ndc = files['A_ndc']
    H = files['H']

    # create an index map
    index_map = []
    count = 0
    for i in range(START_IDX_INT, END_IDX_INT):
        for j in range(START_IDX_CONSTR, END_IDX_CONSTR):
            for k in range(START_IDX_FUNC, END_IDX_FUNC):
                for l in range(START_IDX_SAMPLE, END_IDX_SAMPLE):
                    index_map.append(np.array([i, j, k, l]))
                    count += 1
    index_map = np.array(index_map)

    # create the constraint dictionary
    constraint_dict = {
        0: A_nn,
        1: A_nd,
        2: A_ndc
    }

    # perform cholesky transform -- we use the true smear means
    Sigma = np.diag(smear_means)
    L_chol = np.linalg.cholesky(a=Sigma)
    L_chol_inv = np.linalg.inv(L_chol)

    if CORRECT_K_SPEC:
        K_tilde = L_chol_inv @ K
    else:
        K_tilde = L_chol_inv @ K_ansatz

    # transform the data
    data_chol = (L_chol_inv @ data[:NUM_SIMS].T).T  # NUM_SIMS x (number of true bins)

    # optimize the intervals
    START = time()
    intervals = parallel_interval_optimize(
        index_map=index_map,
        constraint_dict=constraint_dict,
        H=H,
        data=data_chol,
        K=K_tilde,
        prior_mean=unfold_means_ansatz,
        alpha=ALPHA,
        verbose=VERBOSE,
        options=OPTIMIZER_OPTIONS
    )
    END = time()
    print('--------------------------------------------------')
    print('Time to optimize: %.4f mins' % ((END - START) / 60))
    # print(intervals)

    # move the intervals into a new array
    intervals_final = np.zeros(shape=(3, 3, NUM_AGG_BINS, NUM_SIMS, 2))

    count = 0
    for i in range(START_IDX_INT, END_IDX_INT):
        for j in range(START_IDX_CONSTR, END_IDX_CONSTR):
            for k in range(START_IDX_FUNC, END_IDX_FUNC):
                for l in range(START_IDX_SAMPLE, END_IDX_SAMPLE):
                    intervals_final[i, j, k, l, :] = intervals[count]
                    count += 1

    # save the above
    np.save(
        file=BASE_PATH + '/steeply_falling_spectrum_simulation/intervals_optimized_ansatz_rank_def_uneven_bin.npy',
        arr=intervals_final
    )