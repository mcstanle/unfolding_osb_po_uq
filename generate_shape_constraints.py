"""
Script to generate the shape constraints used in the steeply falling spectrum
example.

There are three varieties:
1. Non-negativity (nn)
2. monotonicity (decreasing) (mono)
3. convex (con)

We then stack to create two additional matrices:
1. Non-negative + Monotonic
2. Non-negative + Monotonic + Convex

NOTE: these matrices are designed only to work with the 60 true bin scheme used
in the paper.

Author        : Michael Stanley
Created       : 17 Nov 2021
Last Modified : 17 Nov 2021
===============================================================================
"""
import numpy as np

if __name__ == "__main__":

    # positivity, monotonicity (decreasing), and non-negative
    A_nn = np.identity(60)
    A_mono = np.zeros(shape=(59, 60))
    A_con = np.zeros(shape=(58, 60))

    # monotonicity
    for i in range(59):
        A_mono[i, i] = -1
        A_mono[i, i + 1] = 1

    # convexity
    for i in range(58):
        A_con[i, i] = -1
        A_con[i, i + 1] = 2
        A_con[i, i + 2] = -1
        
    # non-negative
    for i in range(60):
        A_nn[i, i] = -1
        
    # create matrix with nn and mono
    A_nd = np.vstack((A_nn, A_mono))

    # create matrix with all three
    A_ndc = np.vstack((A_nn, A_mono, A_con))

    # save the above
    np.savez(
        file='./data/steeply_falling_spectrum/constraint_matrices.npz',
        A_nn=A_nn,
        A_mono=A_mono,
        A_con=A_con,
        A_nd=A_nd,
        A_ndc=A_ndc
    )