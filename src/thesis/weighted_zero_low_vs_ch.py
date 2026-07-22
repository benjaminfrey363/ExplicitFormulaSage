"""
Low-real-part weighted zero sum versus Chirre--Helfgott.

Purpose
-------
This script deliberately does *not* compare the full hybrid bound with the
Chirre--Helfgott (CH) bound.  CH assumes RH up to its truncation height, so it
has no separate high-real-part contribution.  The goal here is narrower:
measure whether the thesis draft's LOW-real-part summation is competitive
with CH's weighted-zero summation.

The clean comparison is obtained in the limit sigma_1 -> 1/2+.  In this
limit every H0-to-T correction in Proposition 40 vanishes, and

    B_low(x,T) = x^(-1/2) C_low(T).

Thus x disappears from the main comparison.  We compare the coefficient
C_low(T) with the CH Proposition 7.7 coefficient C_CH(T) at the SAME height T:

    C_low(T) / C_CH(T).

For T > H0, CH's theorem is not known to apply at that height.  Therefore
C_CH(T) is labelled a "same-height RH benchmark": it is a structural benchmark
that isolates the quality of the low-real-part summation, not a second
applicable theorem.

A second, contextual comparison uses C_CH(H0).  Both sides are then supported
by the known verification height, but the truncation heights differ, so that
ratio mixes the quality of the summation with the cost of increasing T.

Why the refined Proposition 7.3 estimate is available near T = H0
------------------------------------------------------------------
At sigma_1 -> 1/2+, the B_phi(H0,T) term is multiplied by zero.  The only
B_phi term remaining is B_phi(t0,T).  Since t0 is small (default 20000) and
H0 is about 3e12, the condition T >= 3 t0 is automatic even when T is only
slightly above H0.  The seemingly restrictive condition T >= 3 H0 matters
only away from the endpoint, where the H0-to-T correction survives, and in
bounds for the high-real-part range.

Outputs
-------
The script writes:

  output/low_vs_ch_same_height.csv
  output/low_vs_ch_t0_sensitivity.csv
  output/low_vs_ch_same_height_percent.png
  output/low_vs_ch_coefficients.png
  output/low_vs_ch_components.png
  output/low_vs_ch_verified_height_context.png
  output/low_vs_ch_sigma_endpoint_sensitivity.png

Run with:

    sage weighted_zero_low_vs_ch.py

The exact low-zero sums use Sage's optional Odlyzko database. Install it with:

    sage -i database_odlyzko_zeta
"""

from pathlib import Path
import csv

from sage.all import RealField, pi, log, exp, sqrt, cot, zeta
from sage.databases.odlyzko import zeta_zeros

import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

PREC = 100
R = RealField(PREC)
PI = R(pi)
TWO_PI = R(2) * PI
E = exp(R(1))
LOG10 = log(R(10))

# Exact Platt--Trudgian verification height quoted by CH.
H0 = R("3000175332800")

# Worst case for the uniform xi in [-1,1] bound.
XI = R(1)

# Exact zeros are summed up to this height; the analytic tail begins here.
T0 = R("20000")

# Deterministic heights.  No random sweep is needed because the endpoint
# comparison is independent of x.
T_RATIOS = [
    R("1.00000001"),
    R("1.000001"),
    R("1.0001"),
    R("1.001"),
    R("1.01"),
    R("1.1"),
    R("2"),
    R("3"),
    R("10"),
    R("100"),
]

# Cutoffs used to check that the exact-sum/analytic-tail split is reasonable.
T0_VALUES = [R("1000"), R("5000"), R("10000"), R("20000")]

# Values used only for the sigma_1 -> 1/2+ convergence diagnostic.
SIGMA_EPSILONS = [
    R("1e-10"), R("3e-10"), R("1e-9"), R("3e-9"),
    R("1e-8"), R("3e-8"), R("1e-7"), R("3e-7"),
    R("1e-6"), R("3e-6"), R("1e-5"), R("3e-5"),
    R("1e-4"), R("3e-4"), R("1e-3"), R("3e-3"),
    R("1e-2"),
]
SIGMA_DIAGNOSTIC_LOG10_X = [R("20"), R("100"), R("1000")]
SIGMA_DIAGNOSTIC_T_RATIO = R("1.01")

OUTPUT_DIR = Path("output")


# =============================================================================
# CH CONSTANTS AND WEIGHTS
# =============================================================================


def _compute_ch_constant(k, terms=50):
    """Compute C_1 or C_2 using the exponentially convergent rearrangement."""
    if k == 1:
        base = R(3) / R(2) - R(2) * log(R(2))
    elif k == 2:
        base = R(7) / R(4) - PI**2 / R(6)
    else:
        raise ValueError("Only C1 and C2 are required")

    correction = R(0)
    for n in range(1, terms + 1):
        m = R(2 * n)
        second_difference = (
            R(1) / m**k
            - R(2) / (m + R(1))**k
            + R(1) / (m + R(2))**k
        )
        correction += (R(zeta(2 * n)) - R(1)) * second_difference
    return base + correction


C1 = _compute_ch_constant(1)
C2 = _compute_ch_constant(2)


def F_weight(u):
    r"""Stable evaluation of F(u)=1/pi-(1-u)cot(pi(1-u)), 0<u<=1."""
    u = R(u)
    if not (R(0) < u <= R(1)):
        raise ValueError("Require 0 < u <= 1")

    # cot(pi(1-u)) = -cot(pi u).  This form is stable for the tiny u values
    # occurring when gamma <= t0 and T is near H0.
    if u <= R("0.5"):
        return R(1) / PI + (R(1) - u) * cot(PI * u)

    v = R(1) - u
    if abs(v) < R("1e-12"):
        return PI * v**2 / R(3) + PI**3 * v**4 / R(45)
    return R(1) / PI - v * cot(PI * v)


def phi_weight(t, T, xi=XI):
    r"""phi_{T,xi}(t)=|F(t/T)+xi(1-t/T)i|."""
    t = R(t)
    T = R(T)
    xi = R(xi)
    if not (R(0) < t <= T):
        raise ValueError("Require 0 < t <= T")
    if abs(xi) > R(1):
        raise ValueError("Require |xi| <= 1")
    u = t / T
    return sqrt(F_weight(u)**2 + xi**2 * (R(1) - u)**2)


# =============================================================================
# EXACT LOW ZEROS
# =============================================================================


def load_zero_ordinates(max_t0):
    """Load all Odlyzko ordinates gamma <= max_t0."""
    try:
        database = zeta_zeros()
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Could not load the Odlyzko database. Run "
            "`sage -i database_odlyzko_zeta`."
        ) from exc

    zeros = []
    for gamma in database:
        gamma = R(gamma)
        if gamma > max_t0:
            break
        zeros.append(gamma)

    if not zeros:
        raise RuntimeError("No zeta zero ordinates were loaded")
    if zeros[-1] <= max_t0 and len(zeros) == len(database):
        raise RuntimeError("Requested t0 exceeds the installed database range")
    return zeros


def exact_low_sums(zero_ordinates, t0, T, xi=XI):
    r"""Return S_low(t0;T,xi), S_1(t0), and the number of included zeros."""
    t0 = R(t0)
    slow = R(0)
    s1 = R(0)
    count = 0
    for gamma in zero_ordinates:
        if gamma > t0:
            break
        slow += phi_weight(gamma, T, xi)
        s1 += R(1) / gamma**2
        count += 1
    return slow, s1, count


# =============================================================================
# ANALYTIC TAIL BOUNDS
# =============================================================================


def B_phi(U, T, refined=True):
    r"""Equation (143), using equation (148) when U>=2*pi*e and T>=3U."""
    U = R(U)
    T = R(T)
    if U < TWO_PI * E or T < U:
        raise ValueError("Require 2*pi*e <= U <= T")

    log_T = log(T / TWO_PI)
    log_U = log(U / TWO_PI)

    main = T / (R(4) * PI**2) * (log_T**2 - log_U**2)
    refined_used = bool(refined and T >= R(3) * U)
    if refined_used:
        main -= T / (R(4) * PI**2) * (
            R(2) * C1 * log(E * T / TWO_PI) - R(2) * C2
        )

    endpoint = (
        (R(2) * log(U) / R(5) + R(4))
        * (T / (PI * U) + PI * U / (R(2) * T))
    )
    variation = R(1) / R(5) * (T / (PI * U) - R(1) / PI)

    return {
        "total": main + endpoint + variation,
        "main": main,
        "endpoint": endpoint,
        "variation": variation,
        "refined_used": refined_used,
    }


def B1(U, T):
    U = R(U)
    T = R(T)
    if not (R(0) < U <= T):
        raise ValueError("Require 0 < U <= T")
    return (
        R(1) / TWO_PI
        * ((log(U / TWO_PI) + R(1)) / U
           - (log(T / TWO_PI) + R(1)) / T)
        + (R(2) * log(U) / R(5) + R(4)) / U**2
        + R(1) / R(10) * (R(1) / U**2 - R(1) / T**2)
    )


def B2(U, T):
    U = R(U)
    T = R(T)
    if not (R(0) < U <= T):
        raise ValueError("Require 0 < U <= T")
    return (
        T / TWO_PI * log(T / (TWO_PI * E))
        - U / TWO_PI * log(U / (TWO_PI * E))
        + log(T) / R(5) + log(U) / R(5) + R(4)
    )


def N_plus(T):
    T = R(T)
    return T / TWO_PI * log(T / TWO_PI)


# =============================================================================
# THE TWO COMPARISON COEFFICIENTS
# =============================================================================


def ch_coefficient(T):
    r"""CH Proposition 7.7 coefficient multiplying x^(-1/2)."""
    T = R(T)
    y = log(T / TWO_PI)
    return y**2 / TWO_PI - R("1.01") * y / (R(6) * PI)


def low_endpoint_components(zero_ordinates, t0, T, xi=XI, refined=True):
    r"""
    Components of C_low(T) in the limit sigma_1 -> 1/2+:

      B_low(x,T) = x^(-1/2) C_low(T).

    All B_phi(H0,T), B1(H0,T), and B2(H0,T) corrections vanish here.
    """
    slow, s1, count = exact_low_sums(zero_ordinates, t0, T, xi)
    bphi = B_phi(t0, T, refined=refined)

    weight = TWO_PI / T * (slow + bphi["total"])
    reciprocal = s1 + B1(t0, T)
    approximation_error = R("4.78") * PI / T**2 * N_plus(T)
    total = weight + reciprocal + approximation_error

    return {
        "weight": weight,
        "reciprocal": reciprocal,
        "approximation_error": approximation_error,
        "total": total,
        "slow": slow,
        "s1": s1,
        "zero_count": count,
        "bphi_total": bphi["total"],
        "bphi_refined_used": bphi["refined_used"],
    }


def low_finite_sigma_normalized(
    zero_ordinates, t0, T, log10_x, epsilon, xi=XI
):
    r"""
    Return sqrt(x) times Proposition 40 with sigma_1=1/2+epsilon.

    This is a diagnostic only.  It shows convergence to the endpoint and
    where the B_phi(H0,T) toggle re-enters the calculation.
    """
    T = R(T)
    epsilon = R(epsilon)
    sigma1 = R("0.5") + epsilon
    L = R(log10_x) * LOG10
    q = exp(epsilon * L)  # sqrt(x) * x^(sigma_1-1) = x^epsilon

    endpoint = low_endpoint_components(zero_ordinates, t0, T, xi, refined=True)

    bphi_H0 = B_phi(H0, T, refined=True)["total"]
    weight_tail = TWO_PI / T * R("0.5") * (q - R(1)) * bphi_H0
    reciprocal_tail = (sigma1 * q - R("0.5")) * B1(H0, T)
    error_tail = PI / T**2 * (
        (R(2) + R("0.78") * sigma1) * q - R("2.39")
    ) * B2(H0, T)

    return endpoint["total"] + weight_tail + reciprocal_tail + error_tail


# =============================================================================
# EVALUATION AND REPORTING
# =============================================================================


def ratio_label(ratio):
    ratio = R(ratio)
    delta = ratio - R(1)
    if R(0) < delta < R("0.001"):
        exponent = int((log(delta) / LOG10).floor())
        mantissa = delta / R(10)**exponent
        return f"1+{float(mantissa):g}e{exponent:+d}"
    return f"{float(ratio):g}"


def evaluate_same_height(zero_ordinates):
    rows = []
    for ratio in T_RATIOS:
        T = ratio * H0
        refined = low_endpoint_components(zero_ordinates, T0, T, refined=True)
        coarse = low_endpoint_components(zero_ordinates, T0, T, refined=False)
        ch_T = ch_coefficient(T)
        ch_H0 = ch_coefficient(H0)

        rows.append({
            "T_ratio": ratio,
            "delta_ratio": ratio - R(1),
            "T": T,
            "low_refined": refined["total"],
            "low_coarse": coarse["total"],
            "low_weight": refined["weight"],
            "low_reciprocal": refined["reciprocal"],
            "low_approximation_error": refined["approximation_error"],
            "CH_same_T": ch_T,
            "CH_H0": ch_H0,
            "same_height_ratio_refined": refined["total"] / ch_T,
            "same_height_ratio_coarse": coarse["total"] / ch_T,
            "verified_height_context_ratio": refined["total"] / ch_H0,
            "Bphi_refined_used": refined["bphi_refined_used"],
            "number_low_zeros": refined["zero_count"],
        })
    return rows


def evaluate_t0_sensitivity(zero_ordinates):
    rows = []
    for ratio in [R("1.00000001"), R("2"), R("10"), R("100")]:
        T = ratio * H0
        for t0 in T0_VALUES:
            result = low_endpoint_components(zero_ordinates, t0, T, refined=True)
            rows.append({
                "T_ratio": ratio,
                "t0": t0,
                "zero_count": result["zero_count"],
                "low_coefficient": result["total"],
                "CH_same_T": ch_coefficient(T),
                "same_height_ratio": result["total"] / ch_coefficient(T),
            })
    return rows


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row[key]) for key in columns})


def print_report(rows, t0_rows):
    print("\n" + "=" * 92)
    print("LOW-REAL-PART SUM VERSUS CH: SAME-HEIGHT BENCHMARK")
    print("=" * 92)
    print(f"H0                         = {H0}")
    print(f"t0                         = {T0}")
    print(f"zeros included             = {rows[0]['number_low_zeros']}")
    print(f"xi                         = {XI} (uniform worst case)")
    print(f"C1                         = {C1}")
    print(f"C2                         = {C2}")
    print("\nCH(T) below is a same-height RH benchmark for T>H0, not an applicable theorem.")
    print("At sigma_1 -> 1/2+, the comparison is independent of x.")

    header = (
        f"{'T/H0':>13} {'C_low refined':>17} {'C_CH(T)':>17} "
        f"{'refined/CH':>13} {'coarse/CH':>12} {'C_low/CH(H0)':>15}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{ratio_label(row['T_ratio']):>13} "
            f"{float(row['low_refined']):17.10f} "
            f"{float(row['CH_same_T']):17.10f} "
            f"{float(row['same_height_ratio_refined']):13.9f} "
            f"{float(row['same_height_ratio_coarse']):12.9f} "
            f"{float(row['verified_height_context_ratio']):15.9f}"
        )

    print("\nCOMPONENTS AT SELECTED HEIGHTS")
    print("-" * 92)
    for row in rows:
        if row["T_ratio"] not in [R("1.00000001"), R("2"), R("10"), R("100")]:
            continue
        total = row["low_refined"]
        print(f"T/H0 = {ratio_label(row['T_ratio'])}")
        print(
            f"  weight term             {float(row['low_weight']):.12f} "
            f"({100*float(row['low_weight']/total):.6f}%)"
        )
        print(
            f"  reciprocal term         {float(row['low_reciprocal']):.12f} "
            f"({100*float(row['low_reciprocal']/total):.6f}%)"
        )
        print(
            f"  approximation error     {float(row['low_approximation_error']):.12e} "
            f"({100*float(row['low_approximation_error']/total):.6e}%)"
        )

    grouped = {}
    for row in t0_rows:
        grouped.setdefault(row["T_ratio"], []).append(row)
    print("\nT0 SENSITIVITY")
    print("-" * 92)
    for ratio, group in grouped.items():
        values = [row["same_height_ratio"] for row in group]
        spread = max(values) - min(values)
        print(
            f"T/H0={ratio_label(ratio):>8}: ratio range "
            f"[{float(min(values)):.9f}, {float(max(values)):.9f}], "
            f"spread={float(spread):.3e}"
        )


# =============================================================================
# PLOTS
# =============================================================================


def _save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote plot: {path}")


def plot_same_height_percent(rows, path):
    x = [float(row["delta_ratio"]) for row in rows]
    refined = [100 * float(row["same_height_ratio_refined"] - R(1)) for row in rows]
    coarse = [100 * float(row["same_height_ratio_coarse"] - R(1)) for row in rows]

    plt.figure(figsize=(9.2, 5.8))
    plt.plot(x, refined, marker="o", linewidth=2.2, label="Refined low-real-part bound")
    plt.plot(x, coarse, marker="s", linewidth=1.8, label="Coarse low-real-part bound")
    plt.axhline(0, linestyle="--", linewidth=1.4, label="Equal to CH benchmark")
    plt.xscale("log")
    plt.xlabel(r"Relative extension $(T-H_0)/H_0$")
    plt.ylabel("Difference from same-height CH benchmark (%)")
    plt.title("Low-real-part summation versus CH at the same height")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save_figure(path)


def plot_coefficients(rows, path):
    x = [float(row["delta_ratio"]) for row in rows]
    low = [float(row["low_refined"]) for row in rows]
    ch = [float(row["CH_same_T"]) for row in rows]

    plt.figure(figsize=(9.2, 5.8))
    plt.plot(x, ch, marker="o", linewidth=2.2, label="CH coefficient at height T")
    plt.plot(x, low, marker="s", linewidth=2.0, label="Refined low-real-part coefficient")
    plt.xscale("log")
    plt.xlabel(r"Relative extension $(T-H_0)/H_0$")
    plt.ylabel(r"Coefficient multiplying $x^{-1/2}$")
    plt.title("Absolute size of the weighted-zero coefficients")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save_figure(path)


def plot_components(rows, path):
    selected = [
        row for row in rows
        if row["T_ratio"] in [R("1.00000001"), R("2"), R("10"), R("100")]
    ]
    labels = [ratio_label(row["T_ratio"]) for row in selected]
    weight = [100 * float(row["low_weight"] / row["low_refined"]) for row in selected]
    reciprocal = [100 * float(row["low_reciprocal"] / row["low_refined"]) for row in selected]
    error = [100 * float(row["low_approximation_error"] / row["low_refined"]) for row in selected]

    positions = list(range(len(selected)))
    width = 0.24
    plt.figure(figsize=(9.2, 5.8))
    plt.bar([p - width for p in positions], weight, width=width, label="Simplified weight sum")
    plt.bar(positions, reciprocal, width=width, label=r"$1/\gamma^2$ error")
    plt.bar([p + width for p in positions], error, width=width, label="Remaining weight error")
    plt.xticks(positions, labels)
    plt.yscale("log")
    plt.xlabel(r"Truncation-height ratio $T/H_0$")
    plt.ylabel("Share of refined low coefficient (%)")
    plt.title("Composition of the low-real-part bound")
    plt.legend()
    plt.grid(True, axis="y", which="both", alpha=0.25)
    _save_figure(path)


def plot_verified_height_context(rows, path):
    x = [float(row["T_ratio"]) for row in rows]
    ratio = [float(row["verified_height_context_ratio"]) for row in rows]

    plt.figure(figsize=(9.2, 5.8))
    plt.plot(x, ratio, marker="o", linewidth=2.2)
    plt.axhline(1, linestyle="--", linewidth=1.4)
    plt.xscale("log")
    plt.xlabel(r"Truncation-height ratio $T/H_0$")
    plt.ylabel(r"$C_{\rm low}(T)/C_{\rm CH}(H_0)$")
    plt.title("Different-height comparison (context only)")
    plt.text(
        0.02, 0.96,
        "This mixes summation quality with the cost of increasing T.",
        transform=plt.gca().transAxes,
        va="top",
    )
    plt.grid(True, alpha=0.25)
    _save_figure(path)


def plot_sigma_endpoint_sensitivity(zero_ordinates, path):
    T = SIGMA_DIAGNOSTIC_T_RATIO * H0
    endpoint = low_endpoint_components(zero_ordinates, T0, T, refined=True)["total"]
    x_values = [float(epsilon) for epsilon in SIGMA_EPSILONS]

    plt.figure(figsize=(9.2, 5.8))
    for log10_x in SIGMA_DIAGNOSTIC_LOG10_X:
        differences = []
        for epsilon in SIGMA_EPSILONS:
            finite = low_finite_sigma_normalized(
                zero_ordinates, T0, T, log10_x, epsilon
            )
            differences.append(100 * float(finite / endpoint - R(1)))
        plt.plot(
            x_values,
            differences,
            marker="o",
            linewidth=1.8,
            label=rf"$\log_{{10}}x={float(log10_x):g}$",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\sigma_1-1/2$")
    plt.ylabel("Excess over endpoint coefficient (%)")
    plt.title(r"Convergence to the limit $\sigma_1\to 1/2^+$ at $T=1.01H_0$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)
    _save_figure(path)


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Odlyzko zero ordinates...")
    zeros = load_zero_ordinates(max(T0_VALUES))
    print(f"Loaded {len(zeros)} zeros with gamma <= {max(T0_VALUES)}")

    rows = evaluate_same_height(zeros)
    t0_rows = evaluate_t0_sensitivity(zeros)
    print_report(rows, t0_rows)

    same_height_csv = OUTPUT_DIR / "low_vs_ch_same_height.csv"
    write_csv(
        same_height_csv,
        rows,
        [
            "T_ratio", "delta_ratio", "T",
            "low_refined", "low_coarse",
            "low_weight", "low_reciprocal", "low_approximation_error",
            "CH_same_T", "CH_H0",
            "same_height_ratio_refined", "same_height_ratio_coarse",
            "verified_height_context_ratio",
            "Bphi_refined_used", "number_low_zeros",
        ],
    )
    print(f"Wrote CSV: {same_height_csv}")

    sensitivity_csv = OUTPUT_DIR / "low_vs_ch_t0_sensitivity.csv"
    write_csv(
        sensitivity_csv,
        t0_rows,
        [
            "T_ratio", "t0", "zero_count", "low_coefficient",
            "CH_same_T", "same_height_ratio",
        ],
    )
    print(f"Wrote CSV: {sensitivity_csv}")

    plot_same_height_percent(
        rows, OUTPUT_DIR / "low_vs_ch_same_height_percent.png"
    )
    plot_coefficients(
        rows, OUTPUT_DIR / "low_vs_ch_coefficients.png"
    )
    plot_components(
        rows, OUTPUT_DIR / "low_vs_ch_components.png"
    )
    plot_verified_height_context(
        rows, OUTPUT_DIR / "low_vs_ch_verified_height_context.png"
    )
    plot_sigma_endpoint_sensitivity(
        zeros, OUTPUT_DIR / "low_vs_ch_sigma_endpoint_sensitivity.png"
    )


if __name__ == "__main__":
    main()
