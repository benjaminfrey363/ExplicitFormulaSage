"""
perron_plots.sage

Plotting experiments for classical Perron's formula.

This script produces convergence/error plots for:

1. F(s) = zeta(s), where Perron's formula approximates floor(x).
2. F(s) = -zeta'(s)/zeta(s), where Perron's formula approximates psi(x).

Run from the repo root with:

    sage src/perron/perron_plots.sage
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Load project functions
# ============================================================

load("src/primecounting/chebyshev.py")


# ============================================================
# Perron integral machinery
# ============================================================

def perron_type_integral(F, K, x, c=2, T=50):
    """
    Numerically compute

        (1 / 2 pi i) int_{c-iT}^{c+iT} F(s) x^s K(s) ds.

    Parametrizing s = c + it gives

        (1 / 2 pi) int_{-T}^{T} F(c+it) x^(c+it) K(c+it) dt.
    """
    def integrand(t):
        s = c + I*t
        return F(s) * x**s * K(s)

    real_part = numerical_integral(lambda t: real(integrand(t)), -T, T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), -T, T)[0]

    return (real_part + I*imag_part) / (2*pi)


def K_classical(s):
    """
    Classical Perron kernel.
    """
    return 1/s


def perron_integral(F, x, c=2, T=50):
    """
    Classical Perron integral.
    """
    return perron_type_integral(F, K_classical, x=x, c=c, T=T)


# ============================================================
# Dirichlet series examples
# ============================================================

def F_zeta(s):
    """
    F(s) = zeta(s).

    Perron should recover floor(x).
    """
    return zeta(s)


def minus_zeta_prime_over_zeta(s, h=1e-6):
    """
    Numerically approximate

        -zeta'(s)/zeta(s).

    This is good enough for exploratory plotting.
    """
    zeta_prime_approx = (zeta(s + h) - zeta(s - h)) / (2*h)
    return -zeta_prime_approx / zeta(s)


def exact_sum_for_zeta(x):
    """
    Exact target for F(s)=zeta(s).
    """
    return floor(x)


def exact_sum_for_psi(x):
    """
    Exact target for F(s)=-zeta'(s)/zeta(s).
    """
    return chebyshev_psi(x)


# ============================================================
# Convergence data
# ============================================================

def convergence_data(F, exact_func, x, c=2, T_values=None):
    """
    Compute approximation and error data as T varies.

    Returns a list of dictionaries.
    """
    if T_values is None:
        T_values = [5, 10, 15, 20, 30, 40, 50, 75, 100]

    exact = exact_func(x)
    rows = []

    for T in T_values:
        print(f"Computing x={x}, T={T}...")
        approx = perron_integral(F, x=x, c=c, T=T)

        rows.append({
            "T": T,
            "approx": real(approx),
            "imaginary": imag(approx),
            "exact": exact,
            "error": real(approx) - exact,
            "absolute_error": abs(real(approx) - exact),
        })

    return rows


# ============================================================
# Plot helpers
# ============================================================

def ensure_output_dir(path="outputs/perron_plots"):
    """
    Create output directory if needed.
    """
    os.makedirs(path, exist_ok=True)
    return path


def to_float_list(values):
    """
    Convert Sage numerical values to Python floats.
    """
    return [float(N(v)) for v in values]


def plot_approximation_vs_T(rows, title, output_path):
    """
    Plot Perron approximation versus T, with the exact value shown.
    """
    T_values = [row["T"] for row in rows]
    approximations = to_float_list([row["approx"] for row in rows])
    exact_values = to_float_list([row["exact"] for row in rows])

    plt.figure()
    plt.plot(T_values, approximations, marker="o", label="Perron approximation")
    plt.plot(T_values, exact_values, linestyle="--", label="Exact value")
    plt.xlabel("Truncation height T")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def plot_error_vs_T(rows, title, output_path):
    """
    Plot signed error versus T.
    """
    T_values = [row["T"] for row in rows]
    errors = to_float_list([row["error"] for row in rows])

    plt.figure()
    plt.plot(T_values, errors, marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Truncation height T")
    plt.ylabel("Signed error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def plot_absolute_error_vs_T(rows, title, output_path):
    """
    Plot absolute error versus T.
    """
    T_values = [row["T"] for row in rows]
    errors = to_float_list([row["absolute_error"] for row in rows])

    plt.figure()
    plt.plot(T_values, errors, marker="o")
    plt.xlabel("Truncation height T")
    plt.ylabel("Absolute error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


def plot_imaginary_part_vs_T(rows, title, output_path):
    """
    Plot imaginary part of the computed Perron integral.

    The true target is real, so this is a numerical sanity check.
    """
    T_values = [row["T"] for row in rows]
    imaginary_parts = to_float_list([row["imaginary"] for row in rows])

    plt.figure()
    plt.plot(T_values, imaginary_parts, marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Truncation height T")
    plt.ylabel("Imaginary part")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved {output_path}")


# ============================================================
# Main plotting routines
# ============================================================

def make_zeta_plots(x=25.7, c=2):
    """
    Make plots for F(s)=zeta(s), where Perron approximates floor(x).
    """
    output_dir = ensure_output_dir()

    T_values = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    rows = convergence_data(
        F_zeta,
        exact_sum_for_zeta,
        x=x,
        c=c,
        T_values=T_values,
    )

    plot_approximation_vs_T(
        rows,
        title=f"Perron approximation to floor({x})",
        output_path=f"{output_dir}/zeta_approximation_vs_T.png",
    )

    plot_error_vs_T(
        rows,
        title=f"Signed error for Perron approximation to floor({x})",
        output_path=f"{output_dir}/zeta_error_vs_T.png",
    )

    plot_absolute_error_vs_T(
        rows,
        title=f"Absolute error for Perron approximation to floor({x})",
        output_path=f"{output_dir}/zeta_absolute_error_vs_T.png",
    )

    plot_imaginary_part_vs_T(
        rows,
        title=f"Imaginary part check for F(s)=zeta(s)",
        output_path=f"{output_dir}/zeta_imaginary_part_vs_T.png",
    )


def make_psi_plots(x=100.5, c=2):
    """
    Make plots for F(s)=-zeta'(s)/zeta(s), where Perron approximates psi(x).
    """
    output_dir = ensure_output_dir()

    # Keep this modest at first because numerical differentiation makes
    # each integral more expensive.
    T_values = [5, 10, 15, 20, 30, 40, 50]

    rows = convergence_data(
        minus_zeta_prime_over_zeta,
        exact_sum_for_psi,
        x=x,
        c=c,
        T_values=T_values,
    )

    plot_approximation_vs_T(
        rows,
        title=f"Perron approximation to psi({x})",
        output_path=f"{output_dir}/psi_approximation_vs_T.png",
    )

    plot_error_vs_T(
        rows,
        title=f"Signed error for Perron approximation to psi({x})",
        output_path=f"{output_dir}/psi_error_vs_T.png",
    )

    plot_absolute_error_vs_T(
        rows,
        title=f"Absolute error for Perron approximation to psi({x})",
        output_path=f"{output_dir}/psi_absolute_error_vs_T.png",
    )

    plot_imaginary_part_vs_T(
        rows,
        title=f"Imaginary part check for F(s)=-zeta'(s)/zeta(s)",
        output_path=f"{output_dir}/psi_imaginary_part_vs_T.png",
    )


def make_all_plots():
    """
    Generate all plots.
    """
    print("=" * 70)
    print("Making zeta Perron plots")
    print("=" * 70)
    make_zeta_plots()

    print()
    print("=" * 70)
    print("Making psi Perron plots")
    print("=" * 70)
    make_psi_plots()


# ============================================================
# Run script
# ============================================================

make_all_plots()