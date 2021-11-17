"""
Script generating the bin aggregating functions for both the deconvolution
and steeply falling spectrum sections.

We use the convention "H" to match the lower-case "h" used to represent the
linear functionals in the paper. Each row of "H" is one "h".

Author        : Michael Stanley
Created       : 01 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
import numpy as np

if __name__ == "__main__":

    # deconvolution example ----------
    # full-rank, i.e., 40x40
    H = np.zeros(shape=(10, 40))
    for i in range(10):
        H[i, (4 * i):(4 * (i + 1))] = 1

    np.save(file='./functionals/H_deconvolution.npy', arr=H)

    # Rank-Deficient, e.g., 40x80
    H_80 = np.zeros(shape=(10, 80))
    for i in range(10):
        H_80[i, (8 * i):(8 * (i + 1))] = 1
    np.save(file='./functionals/H_80_deconvolution.npy', arr=H_80)

    H_160 = np.zeros(shape=(10, 160))
    for i in range(10):
        H_160[i, (16 * i):(16 * (i + 1))] = 1
    np.save(file='./functionals/H_160_deconvolution.npy', arr=H_160)

    H_320 = np.zeros(shape=(10, 320))
    for i in range(10):
        H_320[i, (32 * i):(32 * (i + 1))] = 1
    np.save(file='./functionals/H_320_deconvolution.npy', arr=H_320)

    # steeply falling spectrum example ---------- 
    H = np.zeros(shape=(10, 60))

    true_func_endpoints = np.square(np.linspace(np.sqrt(400), np.sqrt(1000), num=11))

    # find the closest endpoints for each fine bin
    true_grid = np.linspace(start=400, stop=1000, num=60 + 1)
    close_func_endpoints = np.zeros(11)
    for i in range(10):
        ep_dists = np.abs(true_grid - true_func_endpoints[i])
        argmin = np.argmin(ep_dists)
        close_func_endpoints[i] = true_grid[argmin]
        
    close_func_endpoints[-1] = 1000

    close_ep_current = 1
    functional_current = 0
    for i in range(60):
        if true_grid[i] < close_func_endpoints[close_ep_current]:
            H[functional_current, i] = 1
        else:
            functional_current += 1
            close_ep_current += 1
            H[functional_current, i] = 1

    np.save(file='./functionals/H_steeply_falling_spectrum.npy', arr=H)
    np.save(file='./functionals/sfs_close_func_endpoints.npy', arr=close_func_endpoints)