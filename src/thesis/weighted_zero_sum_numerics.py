"""
Numerics for the weighted zero-sum bound in the thesis draft.

The script compares three quantities:

1. HYBRID:
   The full ZFR/splitting bound B_{<sigma_1} + B_{>=sigma_1}
   from Propositions 38--39 and Corollary 41 of weighted_zero_sum.

2. CH AT H0:
   Chirre--Helfgott Proposition 7.7 used at the verified RH height H0,

       x^(-1/2) * [log(T/(2*pi))^2/(2*pi)
                    - 1.01*log(T/(2*pi))/(6*pi)],

   with T = H0.

3. CH AT T (IDEAL BENCHMARK):
   The same Chirre--Helfgott estimate at the larger truncation height T.
   This is only a benchmark unless RH is verified up to T.

The script also compares the weighted-zero contribution together with the
leading truncation term pi/T.  Thus it records whether

    pi/T + hybrid_zero_bound < pi/H0 + ch_zero_bound(x, H0),

ignoring the other terms in the full explicit formula.

Run with:

    sage weighted_zero_sum_numerics.py

The exact low-zero sums use Sage's optional Odlyzko database.  Install it with:

    sage -i database_odlyzko_zeta

The default sweep is deliberately concentrated near T = H0, because that is
where increasing T has the best chance to improve the CH truncation term
without paying too much for the unverified zero range.
"""

from sage.all import RealField, pi, log, exp, sqrt, cot
from sage.databases.odlyzko import zeta_zeros

import csv
import random
from pathlib import Path


"""
CONSTANTS
"""

PREC = 100
R = RealField(prec=PREC)
PI = R(pi)
E = exp(R(1))
TWO_PI = R(2) * PI
LOG_10 = log(R(10))

# Exact Platt--Trudgian verification height quoted by Chirre--Helfgott.
DEFAULT_H0 = R("3000175332800")

# Provisional classical zero-free-region constant used in the draft context.
# Keep this as an explicit argument so that a different theorem/constant can
# be substituted without changing any formulas below.
DEFAULT_ZFR_R = R("5.558691")

# Mirrors the low-zero cutoff used in the proof of CH Proposition 7.7.
DEFAULT_T0 = R("20000")

# The uniform bound is worst at |xi| = 1; the sign does not matter because the
# simplified real-axis weight contains xi only through xi^2.
DEFAULT_XI = R(1)


"""
VALIDATION
"""


def _validate_height(name, value):
    value = R(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_parameters(L, H0, T, sigma1, t0, zfr_R, xi):
    L = R(L)
    H0 = _validate_height("H0", H0)
    T = _validate_height("T", T)
    sigma1 = R(sigma1)
    t0 = _validate_height("t0", t0)
    zfr_R = _validate_height("zfr_R", zfr_R)
    xi = R(xi)

    if T <= H0:
        raise ValueError("Require T > H0")
    if not (R("0.5") < sigma1 < R(1)):
        raise ValueError("Require 1/2 < sigma1 < 1")
    if abs(xi) > 1:
        raise ValueError("Require -1 <= xi <= 1")
    if t0 < TWO_PI * E:
        raise ValueError("Require t0 >= 2*pi*e")
    if t0 >= H0:
        raise ValueError("Require t0 < H0")
    if L <= log(T):
        raise ValueError("Require x > T, equivalently log(x) > log(T)")

    return L, H0, T, sigma1, t0, zfr_R, xi


"""
CHIRRE--HELFGOTT REAL-AXIS WEIGHT AND EXACT LOW-ZERO SUMS
"""


def F_weight(u):
    r"""
    Compute

        F(u) = 1/pi - (1-u) cot(pi(1-u)),   0 < u <= 1.

    A short series is used near u = 1 to avoid subtractive cancellation.
    """
    u = R(u)
    if u <= 0 or u > 1:
        raise ValueError("F_weight requires 0 < u <= 1")

    v = R(1) - u

    if abs(v) < R("1e-12"):
        # From z*cot(z) = 1 - z^2/3 - z^4/45 - 2z^6/945 - ...
        return (
            PI * v**2 / R(3)
            + PI**3 * v**4 / R(45)
            + R(2) * PI**5 * v**6 / R(945)
        )

    # This form is stable when u is small; the original form is stable away
    # from the endpoint u = 1.
    if u <= R("0.5"):
        return R(1) / PI + (R(1) - u) * cot(PI * u)

    return R(1) / PI - v * cot(PI * v)


def simplified_weight(t, T, xi=DEFAULT_XI):
    r"""
    Compute

        phi_{T,xi}(t) = |F(t/T) + xi(1-t/T)i|.
    """
    t = R(t)
    T = _validate_height("T", T)
    xi = R(xi)

    if t <= 0 or t > T:
        raise ValueError("Require 0 < t <= T")
    if abs(xi) > 1:
        raise ValueError("Require -1 <= xi <= 1")

    u = t / T
    return sqrt(F_weight(u)**2 + xi**2 * (R(1) - u)**2)


def load_zero_ordinates(t0):
    """
    Return all Odlyzko zero ordinates gamma <= t0.

    The optional Sage database contains the first 2,001,052 ordinates.  The
    function stops as soon as it passes t0, so subsequent computations use
    only the required prefix.
    """
    t0 = _validate_height("t0", t0)

    try:
        all_zeros = zeta_zeros()
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Could not load the Odlyzko zero database. Install it with "
            "`sage -i database_odlyzko_zeta`."
        ) from exc

    selected = []
    for gamma in all_zeros:
        gamma = R(gamma)
        if gamma > t0:
            break
        selected.append(gamma)

    if not selected:
        raise RuntimeError(f"No zero ordinates were found below t0={t0}")

    if selected[-1] <= t0 and len(selected) == len(all_zeros):
        raise RuntimeError(
            "The requested t0 exceeds the range of the installed Odlyzko "
            "database."
        )

    return selected


def exact_low_zero_sums(zero_ordinates, t0, T, xi=DEFAULT_XI):
    r"""
    Compute the exact low-zero quantities

        S_low(t0; T, xi) = sum_{0 < gamma <= t0} phi_{T,xi}(gamma),
        S_1(t0)          = sum_{0 < gamma <= t0} 1/gamma^2.
    """
    t0 = R(t0)
    T = R(T)
    xi = R(xi)

    slow = R(0)
    s1 = R(0)
    count = 0

    for gamma in zero_ordinates:
        gamma = R(gamma)
        if gamma > t0:
            break
        slow += simplified_weight(gamma, T, xi)
        s1 += R(1) / gamma**2
        count += 1

    return {
        "slow": slow,
        "s1": s1,
        "number_low_zeros": count,
    }


"""
AUXILIARY ANALYTIC BOUNDS FROM THE DRAFT
"""


def B_weight(U, T):
    r"""B_phi(U,T) from equation (143)."""
    U = _validate_height("U", U)
    T = _validate_height("T", T)
    if T < U:
        raise ValueError("Require T >= U")

    log_T = log(T / TWO_PI)
    log_U = log(U / TWO_PI)

    main_integral = (
        T / (R(4) * PI**2)
        * (log_T - log_U)
        * (log_T + log_U)
    )

    endpoint = (
        (R(2) * log(U) / R(5) + R(4))
        * (T / (PI * U) + PI * U / (R(2) * T))
    )

    variation = (
        R(1) / R(5)
        * (T / (PI * U) - R(1) / PI)
    )

    return main_integral + endpoint + variation


def B1(U, T):
    r"""B_1(U,T) from equation (155)."""
    U = _validate_height("U", U)
    T = _validate_height("T", T)
    if T < U:
        raise ValueError("Require T >= U")

    integral = (
        R(1) / TWO_PI
        * (
            (log(U / TWO_PI) + R(1)) / U
            - (log(T / TWO_PI) + R(1)) / T
        )
    )

    endpoint = (
        R(1) / U**2
        * (R(2) * log(U) / R(5) + R(4))
    )

    variation = (
        R(1) / R(10)
        * (R(1) / U**2 - R(1) / T**2)
    )

    return integral + endpoint + variation


def B2(U, T):
    r"""B_2(U,T) from equation (158)."""
    U = _validate_height("U", U)
    T = _validate_height("T", T)
    if T < U:
        raise ValueError("Require T >= U")

    return (
        T / TWO_PI * log(T / (TWO_PI * E))
        - U / TWO_PI * log(U / (TWO_PI * E))
        + log(T) / R(5)
        + log(U) / R(5)
        + R(4)
    )


def N_plus_bound(T):
    r"""N^+(T) = T/(2*pi) log(T/(2*pi)) from equation (159)."""
    T = _validate_height("T", T)
    return T / TWO_PI * log(T / TWO_PI)


"""
WEIGHTED ZERO-SUM BOUNDS
"""


def low_real_part_bound(L, H0, T, sigma1, t0, low_sums):
    r"""
    Compute B_{<sigma_1} from Proposition 38, retaining each component.
    """
    L = R(L)
    H0 = R(H0)
    T = R(T)
    sigma1 = R(sigma1)
    t0 = R(t0)

    x_half = exp(-L / R(2))
    x_sigma = exp((sigma1 - R(1)) * L)

    slow = R(low_sums["slow"])
    s1 = R(low_sums["s1"])

    weight_base = (
        TWO_PI / T
        * x_half
        * (slow + B_weight(t0, T))
    )

    weight_tail = (
        TWO_PI / T
        * R("0.5")
        * (x_sigma - x_half)
        * B_weight(H0, T)
    )

    reciprocal_base = x_half * (s1 + B1(t0, T))

    reciprocal_tail = (
        (sigma1 * x_sigma - R("0.5") * x_half)
        * B1(H0, T)
    )

    error_base = (
        R("4.78") * PI / T**2
        * x_half
        * N_plus_bound(T)
    )

    error_tail = (
        PI / T**2
        * (
            (R(2) + R("0.78") * sigma1) * x_sigma
            - R("2.39") * x_half
        )
        * B2(H0, T)
    )

    total = (
        weight_base
        + weight_tail
        + reciprocal_base
        + reciprocal_tail
        + error_base
        + error_tail
    )

    return {
        "low_weight_base": weight_base,
        "low_weight_tail": weight_tail,
        "low_reciprocal_base": reciprocal_base,
        "low_reciprocal_tail": reciprocal_tail,
        "low_error_base": error_base,
        "low_error_tail": error_tail,
        "low_total": total,
    }


def high_real_part_bound(L, H0, T, zfr_R):
    r"""
    Compute B_{>=sigma_1} from Proposition 39.

    The current bound is independent of sigma_1 because the high-real-part
    zero sums are enlarged to include every zero with H0 < gamma < T before
    applying the zero-free region.
    """
    L = R(L)
    H0 = R(H0)
    T = R(T)
    zfr_R = R(zfr_R)

    zfr_factor = exp(-L / (zfr_R * log(T)))

    weight_part = TWO_PI / T * B_weight(H0, T)
    reciprocal_part = R(2) * B1(H0, T)
    error_part = R("5.56") * PI / T**2 * B2(H0, T)

    bracket = weight_part + reciprocal_part + error_part
    total = zfr_factor * bracket

    return {
        "zfr_factor": zfr_factor,
        "high_weight_part": zfr_factor * weight_part,
        "high_reciprocal_part": zfr_factor * reciprocal_part,
        "high_error_part": zfr_factor * error_part,
        "high_bracket": bracket,
        "high_total": total,
    }


def hybrid_zero_bound(L, H0, T, sigma1, t0, zfr_R, low_sums):
    """Return the full Corollary 41 bound and its components."""
    low = low_real_part_bound(L, H0, T, sigma1, t0, low_sums)
    high = high_real_part_bound(L, H0, T, zfr_R)

    return {
        **low,
        **high,
        "hybrid_zero_bound": low["low_total"] + high["high_total"],
    }


def ch_coefficient(T):
    r"""Coefficient in Chirre--Helfgott Proposition 7.7."""
    T = _validate_height("T", T)
    y = log(T / TWO_PI)
    return y**2 / TWO_PI - R("1.01") * y / (R(6) * PI)


def ch_zero_bound(L, T):
    r"""CH Proposition 7.7 after multiplication by x^(-1/2)."""
    L = R(L)
    T = _validate_height("T", T)
    return exp(-L / R(2)) * ch_coefficient(T)


"""
PARAMETER SWEEP
"""


def default_T_ratios():
    """
    Ratios concentrate near 1, where the hybrid approach has its best chance.
    """
    return [
        R("1.00000001"),
        R("1.000001"),
        R("1.0001"),
        R("1.001"),
        R("1.01"),
        R("1.1"),
        R("2"),
        R("10"),
        R("100"),
    ]


def default_log_x_offsets():
    r"""Values of log(x/T), so x = T*exp(offset)."""
    return [
        R("0.001"),
        R("0.01"),
        R("0.1"),
        R("0.5"),
        R("1"),
        R("2"),
        R("5"),
        R("10"),
        R("20"),
        R("40"),
        R("80"),
        R("160"),
        R("320"),
        R("640"),
    ]


def default_sigma_values():
    """Split points, with extra resolution close to the lower endpoint."""
    return [
        R("0.500001"),
        R("0.5001"),
        R("0.501"),
        R("0.51"),
        R("0.55"),
        R("0.6"),
        R("0.7"),
        R("0.8"),
        R("0.9"),
        R("0.95"),
    ]


def boundary_parameters(
    H0=DEFAULT_H0,
    t0=DEFAULT_T0,
    zfr_R=DEFAULT_ZFR_R,
    xi=DEFAULT_XI,
    T_ratios=None,
    log_x_offsets=None,
    sigma_values=None,
):
    """Generate the deterministic core grid."""
    H0 = R(H0)
    t0 = R(t0)
    zfr_R = R(zfr_R)
    xi = R(xi)

    if T_ratios is None:
        T_ratios = default_T_ratios()
    if log_x_offsets is None:
        log_x_offsets = default_log_x_offsets()
    if sigma_values is None:
        sigma_values = default_sigma_values()

    parameters = []

    for ratio in T_ratios:
        ratio = R(ratio)
        if ratio <= 1:
            raise ValueError("Every T/H0 ratio must exceed 1")

        T = H0 * ratio

        for offset in log_x_offsets:
            offset = R(offset)
            if offset <= 0:
                raise ValueError("Every log(x/T) offset must be positive")

            L = log(T) + offset

            for sigma1 in sigma_values:
                parameters.append({
                    "L": L,
                    "H0": H0,
                    "T": T,
                    "T_ratio": ratio,
                    "log_x_offset": offset,
                    "sigma1": R(sigma1),
                    "t0": t0,
                    "zfr_R": zfr_R,
                    "xi": xi,
                    "sample_type": "boundary",
                })

    return parameters


def sample_parameters(
    n_samples,
    H0=DEFAULT_H0,
    t0=DEFAULT_T0,
    zfr_R=DEFAULT_ZFR_R,
    xi=DEFAULT_XI,
    T_ratios=None,
    sigma_values=None,
    seed=2026,
):
    """
    Add random (x,T) slices while keeping T on a discrete cache-friendly grid.

    Every random slice is evaluated at the full split-point grid, so the
    subsequent optimization over sigma_1 is genuine rather than depending on
    a single randomly chosen split point.
    """
    if T_ratios is None:
        T_ratios = default_T_ratios()
    if sigma_values is None:
        sigma_values = default_sigma_values()

    rng = random.Random(seed)
    H0 = R(H0)
    t0 = R(t0)
    zfr_R = R(zfr_R)
    xi = R(xi)
    ratios = [R(value) for value in T_ratios]
    sigma_values = [R(value) for value in sigma_values]

    parameters = []

    for _ in range(n_samples):
        ratio = ratios[rng.randrange(len(ratios))]
        T = H0 * ratio

        # log(x/T): log-uniform from 10^-3 through 10^3.
        u = R(str(rng.random()))
        offset = exp(log(R("1e-3")) + u * log(R("1e6")))

        for sigma1 in sigma_values:
            parameters.append({
                "L": log(T) + offset,
                "H0": H0,
                "T": T,
                "T_ratio": ratio,
                "log_x_offset": offset,
                "sigma1": sigma1,
                "t0": t0,
                "zfr_R": zfr_R,
                "xi": xi,
                "sample_type": "random",
            })

    return parameters



"""
SAMPLE EVALUATION
"""


def evaluate_point(params, low_sums):
    """Evaluate one parameter point and retain all analytic components."""
    L, H0, T, sigma1, t0, zfr_R, xi = _validate_parameters(
        params["L"],
        params["H0"],
        params["T"],
        params["sigma1"],
        params["t0"],
        params["zfr_R"],
        params["xi"],
    )

    hybrid = hybrid_zero_bound(
        L=L,
        H0=H0,
        T=T,
        sigma1=sigma1,
        t0=t0,
        zfr_R=zfr_R,
        low_sums=low_sums,
    )

    ch_H0 = ch_zero_bound(L, H0)
    ch_same_T = ch_zero_bound(L, T)

    hybrid_zero = hybrid["hybrid_zero_bound"]

    truncation_H0 = PI / H0
    truncation_T = PI / T

    ch_H0_combined = truncation_H0 + ch_H0
    hybrid_combined = truncation_T + hybrid_zero

    zero_ratio_H0 = hybrid_zero / ch_H0
    zero_ratio_same_T = hybrid_zero / ch_same_T
    combined_ratio = hybrid_combined / ch_H0_combined
    combined_margin = ch_H0_combined - hybrid_combined

    high_fraction = hybrid["high_total"] / hybrid_zero

    return {
        "sample_type": params.get("sample_type", "unspecified"),
        "H0": H0,
        "T": T,
        "T_ratio": T / H0,
        "t0": t0,
        "sigma1": sigma1,
        "xi": xi,
        "zfr_R": zfr_R,
        "L": L,
        "log10_x": L / LOG_10,
        "log_x_offset": L - log(T),
        "number_low_zeros": low_sums["number_low_zeros"],
        "slow": low_sums["slow"],
        "s1": low_sums["s1"],
        **hybrid,
        "ch_H0_zero_bound": ch_H0,
        "ch_same_T_zero_bound": ch_same_T,
        "zero_ratio_H0": zero_ratio_H0,
        "zero_ratio_same_T": zero_ratio_same_T,
        "truncation_H0": truncation_H0,
        "truncation_T": truncation_T,
        "truncation_saving": truncation_H0 - truncation_T,
        "ch_H0_combined": ch_H0_combined,
        "hybrid_combined": hybrid_combined,
        "combined_ratio": combined_ratio,
        "combined_margin": combined_margin,
        "hybrid_combined_wins": combined_margin > 0,
        "high_fraction": high_fraction,
    }


def evaluate_samples(parameters, zero_ordinates, progress_every=250):
    """Evaluate all points, caching exact low-zero sums for each T."""
    results = []
    total = len(parameters)
    low_sum_cache = {}

    for index, params in enumerate(parameters, start=1):
        cache_key = (
            str(R(params["T"])),
            str(R(params["t0"])),
            str(abs(R(params["xi"]))),
        )

        if cache_key not in low_sum_cache:
            low_sum_cache[cache_key] = exact_low_zero_sums(
                zero_ordinates=zero_ordinates,
                t0=params["t0"],
                T=params["T"],
                xi=params["xi"],
            )

        results.append(
            evaluate_point(params, low_sum_cache[cache_key])
        )

        if progress_every and index % progress_every == 0:
            print(f"Evaluated {index}/{total} parameter points")

    return results


"""
OPTIMIZATION AND STATISTICS
"""


def select_best_splits(results):
    """For every (x,H0,T,t0,xi,R) slice, retain the best sigma_1."""
    best = {}

    for row in results:
        key = (
            str(row["L"]),
            str(row["H0"]),
            str(row["T"]),
            str(row["t0"]),
            str(row["xi"]),
            str(row["zfr_R"]),
        )

        if key not in best or row["hybrid_zero_bound"] < best[key]["hybrid_zero_bound"]:
            best[key] = row

    return list(best.values())


def _quantile(values, q):
    if not values:
        raise ValueError("Cannot compute a quantile of empty data")

    q = R(q)
    if q < 0 or q > 1:
        raise ValueError("Require 0 <= q <= 1")

    values = sorted(R(value) for value in values)
    if len(values) == 1:
        return values[0]

    position = q * R(len(values) - 1)
    lower_index = int(position.floor())
    upper_index = int(position.ceil())

    if lower_index == upper_index:
        return values[lower_index]

    weight = position - lower_index
    return (
        (R(1) - weight) * values[lower_index]
        + weight * values[upper_index]
    )


def summarize_results(results):
    if not results:
        raise ValueError("No results to summarize")

    zero_ratios = [row["zero_ratio_H0"] for row in results]
    combined_ratios = [row["combined_ratio"] for row in results]
    high_fractions = [row["high_fraction"] for row in results]
    wins = [row for row in results if row["hybrid_combined_wins"]]

    return {
        "number_points": len(results),
        "number_combined_wins": len(wins),
        "minimum_zero_ratio": min(zero_ratios),
        "median_zero_ratio": _quantile(zero_ratios, R("0.5")),
        "maximum_zero_ratio": max(zero_ratios),
        "minimum_combined_ratio": min(combined_ratios),
        "median_combined_ratio": _quantile(combined_ratios, R("0.5")),
        "maximum_combined_ratio": max(combined_ratios),
        "median_high_fraction": _quantile(high_fractions, R("0.5")),
        "maximum_margin": max(row["combined_margin"] for row in results),
        "minimum_margin": min(row["combined_margin"] for row in results),
    }


"""
REPORTING
"""


def _print_result(row, title=None):
    if title is not None:
        print(title)
        print("-" * len(title))

    print(f"T/H0                      = {float(row['T_ratio']):.10g}")
    print(f"log(x/T)                  = {float(row['log_x_offset']):.10g}")
    print(f"log10(x)                  = {float(row['log10_x']):.10g}")
    print(f"sigma_1                   = {float(row['sigma1']):.10g}")
    print(f"hybrid zero bound         = {float(row['hybrid_zero_bound']):.12e}")
    print(f"  low-real-part bound     = {float(row['low_total']):.12e}")
    print(f"  high-real-part bound    = {float(row['high_total']):.12e}")
    print(f"  high fraction           = {float(row['high_fraction']):.8g}")
    print(f"CH zero bound at H0       = {float(row['ch_H0_zero_bound']):.12e}")
    print(f"CH zero bound at same T   = {float(row['ch_same_T_zero_bound']):.12e}")
    print(f"hybrid / CH(H0)           = {float(row['zero_ratio_H0']):.8g}")
    print(f"hybrid / CH(T), ideal     = {float(row['zero_ratio_same_T']):.8g}")
    print(f"pi/H0 + CH(H0)            = {float(row['ch_H0_combined']):.12e}")
    print(f"pi/T + hybrid             = {float(row['hybrid_combined']):.12e}")
    print(f"combined ratio            = {float(row['combined_ratio']):.8g}")
    print(f"CH(H0) - hybrid margin    = {float(row['combined_margin']):.12e}")
    print(f"hybrid combined wins      = {row['hybrid_combined_wins']}")
    print()


def report_results(results, number_extreme_points=5):
    optimized = select_best_splits(results)
    summary = summarize_results(optimized)

    print()
    print("=" * 80)
    print("WEIGHTED ZERO-SUM NUMERICAL REPORT")
    print("=" * 80)
    print(f"Full parameter points:        {len(results)}")
    print(f"Optimized (x,T) slices:       {summary['number_points']}")
    print(f"Combined hybrid wins:         {summary['number_combined_wins']}")
    print()
    print("Hybrid zero bound / CH zero bound at H0")
    print("----------------------------------------")
    print(f"Minimum:                     {float(summary['minimum_zero_ratio']):.8g}")
    print(f"Median:                      {float(summary['median_zero_ratio']):.8g}")
    print(f"Maximum:                     {float(summary['maximum_zero_ratio']):.8g}")
    print()
    print("(pi/T + hybrid) / (pi/H0 + CH(H0))")
    print("--------------------------------------")
    print(f"Minimum:                     {float(summary['minimum_combined_ratio']):.8g}")
    print(f"Median:                      {float(summary['median_combined_ratio']):.8g}")
    print(f"Maximum:                     {float(summary['maximum_combined_ratio']):.8g}")
    print(f"Median high-range fraction:  {float(summary['median_high_fraction']):.8g}")

    best = sorted(optimized, key=lambda row: row["combined_ratio"])
    worst = sorted(optimized, key=lambda row: row["combined_ratio"], reverse=True)

    print()
    print("=" * 80)
    print("BEST COMBINED COMPARISONS")
    print("=" * 80)
    for index, row in enumerate(best[:number_extreme_points], start=1):
        _print_result(row, f"Best point {index}")

    print("=" * 80)
    print("WORST COMBINED COMPARISONS")
    print("=" * 80)
    for index, row in enumerate(worst[:number_extreme_points], start=1):
        _print_result(row, f"Worst point {index}")

    report_by_T_ratio(optimized)
    report_split_optimization(optimized)

    return optimized


def report_by_T_ratio(optimized_results):
    print()
    print("BEST RESULT GROUPED BY T/H0")
    print("-" * 80)
    print(
        "ratio          n   min zero ratio   min combined   wins   "
        "best log10(x)   best sigma"
    )

    ratio_map = {str(row["T_ratio"]): row["T_ratio"] for row in optimized_results}
    ratios = sorted(ratio_map.values())

    for ratio in ratios:
        group = [row for row in optimized_results if row["T_ratio"] == ratio]
        best = min(group, key=lambda row: row["combined_ratio"])
        wins = sum(1 for row in group if row["hybrid_combined_wins"])

        print(
            f"{float(ratio):<13.8g} "
            f"{len(group):4d} "
            f"{float(min(row['zero_ratio_H0'] for row in group)):16.6g} "
            f"{float(best['combined_ratio']):14.6g} "
            f"{wins:6d} "
            f"{float(best['log10_x']):14.6g} "
            f"{float(best['sigma1']):11.7g}"
        )


def report_split_optimization(optimized_results):
    print()
    print("OPTIMAL SPLIT-POINT FREQUENCIES")
    print("-" * 80)

    counts = {}
    for row in optimized_results:
        key = str(row["sigma1"])
        counts[key] = counts.get(key, 0) + 1

    for sigma_text, count in sorted(counts.items(), key=lambda item: R(item[0])):
        print(f"sigma_1 = {float(R(sigma_text)):<12.8g} count = {count}")


"""
CSV OUTPUT
"""


def write_results_csv(results, filename):
    if not results:
        raise ValueError("No results to write")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                key: str(value) if not isinstance(value, bool) else value
                for key, value in row.items()
            })

    print(f"Wrote CSV: {path}")


"""
PLOTS
"""


def make_plots(optimized_results, output_directory="output"):
    """Save comparison plots for the optimized split point on each slice."""
    import matplotlib.pyplot as plt

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    ratio_map = {str(row["T_ratio"]): row["T_ratio"] for row in optimized_results}
    ratios = sorted(ratio_map.values())

    fig, ax = plt.subplots()
    for ratio in ratios:
        group = sorted(
            [row for row in optimized_results if row["T_ratio"] == ratio],
            key=lambda row: row["log10_x"],
        )
        ax.plot(
            [float(row["log10_x"]) for row in group],
            [float(row["zero_ratio_H0"]) for row in group],
            marker="o",
            markersize=3,
            label=f"T/H0={float(ratio):.6g}",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("log10(x)")
    ax.set_ylabel("hybrid zero bound / CH zero bound at H0")
    ax.set_title("Weighted zero-sum comparison")
    ax.legend(fontsize="small")
    fig.tight_layout()
    zero_plot = output_path / "weighted_zero_ratio.png"
    fig.savefig(zero_plot, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    for ratio in ratios:
        group = sorted(
            [row for row in optimized_results if row["T_ratio"] == ratio],
            key=lambda row: row["log10_x"],
        )
        ax.plot(
            [float(row["log10_x"]) for row in group],
            [float(row["combined_ratio"]) for row in group],
            marker="o",
            markersize=3,
            label=f"T/H0={float(ratio):.6g}",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("log10(x)")
    ax.set_ylabel("(pi/T + hybrid) / (pi/H0 + CH(H0))")
    ax.set_title("Truncation plus weighted-zero comparison")
    ax.legend(fontsize="small")
    fig.tight_layout()
    combined_plot = output_path / "weighted_zero_combined_ratio.png"
    fig.savefig(combined_plot, dpi=200)
    plt.close(fig)

    print(f"Wrote plot: {zero_plot}")
    print(f"Wrote plot: {combined_plot}")


"""
RUN TEST SUITE
"""


def run_test_suite(
    number_random_samples=500,
    H0=DEFAULT_H0,
    t0=DEFAULT_T0,
    zfr_R=DEFAULT_ZFR_R,
    xi=DEFAULT_XI,
    output_directory="output",
    seed=2026,
    create_plots=True,
):
    print("Loading Odlyzko zero ordinates...")
    zero_ordinates = load_zero_ordinates(t0)
    print(f"Loaded {len(zero_ordinates)} zeros with gamma <= {float(R(t0)):.8g}")

    print("Generating deterministic boundary grid...")
    parameters = boundary_parameters(
        H0=H0,
        t0=t0,
        zfr_R=zfr_R,
        xi=xi,
    )

    print(f"Generating {number_random_samples} random (x,T) slices...")
    parameters.extend(
        sample_parameters(
            n_samples=number_random_samples,
            H0=H0,
            t0=t0,
            zfr_R=zfr_R,
            xi=xi,
            seed=seed,
        )
    )

    print(f"Evaluating {len(parameters)} total parameter points...")
    results = evaluate_samples(
        parameters,
        zero_ordinates=zero_ordinates,
        progress_every=250,
    )

    optimized = report_results(results, number_extreme_points=5)

    output_path = Path(output_directory)
    write_results_csv(
        results,
        output_path / f"weighted_zero_all_seed_{seed}.csv",
    )
    write_results_csv(
        optimized,
        output_path / f"weighted_zero_optimized_seed_{seed}.csv",
    )

    if create_plots:
        make_plots(optimized, output_directory=output_directory)

    return results, optimized


if __name__ == "__main__":
    run_test_suite(
        number_random_samples=500,
        output_directory="output",
        seed=2026,
        create_plots=True,
    )
