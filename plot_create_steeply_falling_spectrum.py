"""
Script generating the plots for section 5. Written in conjunction with
plotting_steeply_falling_spectrum.py.

Recommended use: uncomment the figure you would like to plot.

Author        : Michael Stanley
Created       : 16 Nov 2021
Last Modified : 14 Feb 2022
===============================================================================
"""

from plotting_steeply_falling_spectrum import (
    plot_figure15,
    plot_figure16,
    plot_figure17,
    plot_figure18,
    plot_figure19,
    plot_figure20
)

if __name__ == "__main__":

    # base path for saving
    BASE_PATH = '/Users/mikestanley/Research/strict_bounds/prior_optimized_paper/final_images'

    # toggle figures to plot
    # plot_figure15()
    # plot_figure16()
    # plot_figure17()
    # plot_figure18(BASE_PATH + '/steeply_falling_osb_stark_osb_po_ls_example_intervals_AXIS_LABEL_NOT_SHARED.png')
    # plot_figure19()
    plot_figure20(BASE_PATH + '/steeply_falling_osb_stark_osb_po_expected_len.png')
