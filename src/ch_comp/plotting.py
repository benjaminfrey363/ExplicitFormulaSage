


from ch_comp import odlyzko_wrapper as ow, chebyshev, classical, chef
import numpy as np
from sage.all import  numerical_approx
import matplotlib.pyplot as plt
from pathlib import Path


# Approximate L2 distance between 2 functions on pre-specified range xs
# Take as arg values of fns on this range, y1s and y2s
def _l2dist_approx(y1s, y2s):
    if len(y1s) != len(y2s):
        raise ValueError("Function output arrays must have the same size")
    return sum([ abs(y1 - y2)**2 for (y1,y2) in zip(y1s,y2s)])



"""
_, xs, cefs = classical.compute_cef(5000,2,100,1000)
_, _, chefs = chef.compute_chef(5000,2,100,1000)
psis = [chebyshev.chebyshev_psi(x) for x in xs]

print(numerical_approx(_l2dist_approx(cefs,psis),30))
print(numerical_approx(_l2dist_approx(chefs,psis),30))
"""


def save_psi_comparison_plot(xs, psis, approximations, labels, filename):
    """
    Save and display a psi approximation comparison plot.

    Parameters
    ----------
    xs : array-like
        x-values.
    psis : array-like
        Exact/computed psi(x)-values.
    approximations : list[array-like]
        List of approximation arrays to plot.
    labels : list[str]
        Labels for each approximation.
    filename : str
        Output filename, e.g. "cef_low_overlay.png".

    Returns
    -------
    pathlib.Path
        Path to saved plot.
    """
    output_dir = Path("outputs/ch_comp_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(xs, psis, linewidth=2, label=r"$\psi(x)$")

    for ys, label in zip(approximations, labels):
        ax.plot(xs, ys, linewidth=1.5, label=label)

    ax.set_title("Explicit formula approximation to Chebyshev psi")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\psi(x)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return output_path




def full_test_suite():
    """
    Run a full numerical comparison suite for the classical explicit formula
    (CEF) and Chirre--Helfgott explicit formula (CHEF) approximations to psi(x).

    The suite evaluates both approximations over low, mid, and high x-ranges,
    compares them against the directly computed Chebyshev psi function, and
    saves diagnostic plots and summary tables.

    Ranges
    ------
    low : x in [2, 500] - linear grid

    mid : x in [10^3, 10^4] - linear grid

    high : x in [10^4, 10^6] - log-spaced grip

    Saves
    -----
    cef_low_overlay.png
        Plot of psi(x) and CEF_n(x) for n in [100, 500, 1000, 5000] on the low range.

    chef_low_overlay.png
        Plot of psi(x) and CHEF_n(x) for n in [100, 500, 1000, 5000] on the low range.

    cef_low_error.png, cef_mid_error.png, cef_high_error.png
        Plots of CEF_n(x) - psi(x) over the low, mid, and high ranges.

    chef_low_error.png, chef_mid_error.png, chef_high_error.png
        Plots of CHEF_n(x) - psi(x) over the low, mid, and high ranges.

    cef_high_normalized_error.png
        Plot of (CEF_n(x) - psi(x)) / sqrt(x) over the high range.

    chef_high_normalized_error.png
        Plot of (CHEF_n(x) - psi(x)) / sqrt(x) over the high range.

    cef_vs_chef_difference.png
        Plot of CHEF_n(x) - CEF_n(x) for equal n = n_zeros

    cef_error_summary.csv, chef_error_summary.csv
        Tables containing max absolute error, mean absolute error, RMS error,
        approximate L2 error, and runtime for each truncation parameter.

    runtime_vs_truncation.png
        Runtime comparison as the number of zeros or height cutoff increases.

    Returns
    -------
    dict
        Dictionary containing paths to saved plots and summary tables.
    """

    output_dir = Path("outputs/demo_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # cef_low_overlay.png
    # plot of psi(x) and CEF_n(x) for n in [100, 500, 1000, 5000] on the low range.
    

