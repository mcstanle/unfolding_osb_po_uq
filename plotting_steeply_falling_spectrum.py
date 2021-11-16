"""
Script containing all code to generate the plots for section 5.

This script is written so that each function can plot its respective figure in
isolation from any other figure. This way, one can simply toggle the figures
one wants to plot in the main section of the code.

The convention is that the code to plot figure k can be found in function
called "plot_figure[k]". 

Author        : Michael Stanley
Created       : 16 Nov 2021
Last Modified : 16 Nov 2021
===============================================================================
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from utils import intensity_f, compute_mean_std_width

plt.style.use('seaborn-colorblind')


def plot_figure15(save_loc=None):
    """
    Shows the true and ansatz intensity functions and their binned 

    Parameters:
    -----------
        save_loc (str) : saving location -- note saved, by default

    Returns:
    --------
        None -- makes matplotlib plot
    """