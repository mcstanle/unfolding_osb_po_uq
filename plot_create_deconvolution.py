"""
Script generating the plots for section 4. Written in conjunction with
plot_deconvolution.py.

Recommended use: uncomment the figure you would like to plot.

Author        : Michael Stanley
Created       : 04 Nov 2021
Last Modified : 16 Nov 2021
===============================================================================
"""
from plotting_deconvolution import (
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_figure4,
    plot_figure5,
    plot_figure6,
    plot_figure7,
    plot_figure8,
    plot_figure9,
    plot_figure10,
    plot_figure11,
    plot_figure12,
    plot_figure13,
    plot_figure14
)

if __name__ == "__main__":

    # base path for saving
    BASE_PATH = '/Users/mikestanley/Research/strict_bounds/prior_optimized_paper/final_images'

    # toggle figures to plot
    # plot_figure1(save_loc=BASE_PATH + '/intensity_functions.png')
    # plot_figure2(save_loc=BASE_PATH + '/expected_bin_counts.png')
    # plot_figure3(save_loc=BASE_PATH + '/wide_bin_interval_and_coverage_failure.png')
    # plot_figure4(save_loc=BASE_PATH + '/fine_bin_interval_and_coverage_fix_ls.png')
    # plot_figure5(save_loc=BASE_PATH + '/post_agg_correction_ls_intervals.png')
    # plot_figure6(save_loc=BASE_PATH + '/ls_osb_po_specific_intervals_and_expected_lengths.png')
    # plot_figure7(save_loc=BASE_PATH + '/coverage_guarantees_full_rank_OSB_PO.png')
    # plot_figure8(save_loc=BASE_PATH + '/adversarial_coverage_break_95percent.png')
    # plot_figure9()
    # plot_figure10(save_loc=BASE_PATH + '/adversarial_80bin_expected_lengths.png')
    # plot_figure11(save_loc=BASE_PATH + '/expected_interval_length_full_rank_with_stark_donoho.png')
    # plot_figure12()
    # plot_figure13(save_loc=BASE_PATH + '/expected_interval_width_across_bins_40_80_160_320.png')
    plot_figure14(save_loc=BASE_PATH + '/prior_choice_and_expected_length.png')