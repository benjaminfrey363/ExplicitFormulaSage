
"""
Numerics for generalization of Lemma C5 in Chirre-Helfgott

Let 0 < eta <= e and 0 <= beta <= 1. Write L = log x.
For beta L >= 4 (x >= e^{4/beta}), we have

int_0^infty frac{ux^{-u}}{sqrt{eta^2 + (beta - u)^2}}
    <= 1/(beta L^2) + 2/(beta^2 L^3) + 40/(3 beta^3 L^4)
    + x^{-beta} (beta log 1/eta + (beta L + 1)/(eta L^2))

In application, beta is 1 - Re rho for a nontrivial zero rho of zeta(s).
We will then use a zero free region to prevent beta from being too
close to 0 (or 1).

In application, eta = |t - gamma| where t is height of truncation and
gamma is imaginary part of a zeta zero. this quantity will be strictly
positive by choice of t, but test all other possible values

In application x will be large, so for now at least we'll only test for
large x dependent on beta - x >= e^{4/beta}
"""

from sage.all import RealField, e, log, exp, sqrt, numerical_integral
import csv, random
from pathlib import Path


"""
CONSTANTS
"""

PREC = 80
R = RealField(prec=PREC)





"""
PARAMETER VALIDATION and INTEGRAL/UPPER BOUND COMPUTATION
"""

def _validate_beta(beta):
    beta = R(beta)
    if beta < 0 or beta > 1:
        raise ValueError("beta must satisfy 0 <= beta <= 1")
    return beta
    
def _validate_eta(eta):
    eta = R(eta)
    if eta <= 0 or eta > R(e):
        raise ValueError("eta must satisfy 0 < eta <= e")
    return eta
    
def _validate_x(x, beta):
    x = R(x); beta = R(beta)
    if beta * log(x) < R(4):
        raise ValueError("x must satisfy x >= e^{4/beta}")
    return x

def upper_bound(x, eta, beta):
    """
    Compute the upper bound
        1/(beta L^2) + 2/(beta^2 L^3) + 40/(3 beta^3 L^4)
        + x^{-beta} (beta log 1/eta + (beta L + 1)/(eta L^2))
    Assume that parameters are valid
    """
    L = log(x)

    poly_part = (
        R(1) / (beta * L**2)
        + R(2) / (beta**2 * L**3)
        + R(40) / (R(3) * beta**3 * L**4)
    )

    exp_part = x**(-beta) * (
        beta * log(R(1) / eta)
        + (beta * L + R(1)) / (eta * L**2)
    )

    return poly_part + exp_part



def actual_integral(x, eta, beta, tail_constant=100):
    """
    Compute the actual integral
        int_0^infty frac{ux^{-u}}{sqrt{eta^2 + (beta - u)^2}}.
    Robust refactor to accommodate weird numeric integration problems.
    Rescale integral so that decay always begins at w ~ 1, don't need
    to worry about numeric integral sampling bad points
    """
    L = log(x)

    def f(w):
        w = R(w)
        return (
            w * exp(-w)
            / sqrt(eta**2 + (beta - w / L)**2)
        )

    val, err = numerical_integral(
        f,
        R(0),
        R(tail_constant)
    )

    return R(val) / L**2, R(err) / L**2



"""
PARAMETER SAMPLING FUNCTIONS
"""

def _log_uniform(rng, lower, upper):
    """
    Sample log-uniformly from [lower, upper].
    """
    lower = R(lower)
    upper = R(upper)

    if lower <= 0 or upper <= lower:
        raise ValueError("Require 0 < lower < upper")

    t = R(str(rng.random()))
    return exp(log(lower) + t * (log(upper) - log(lower)))


def _uniform(rng, lower, upper):
    """
    Sample uniformly from [lower, upper].
    """
    lower = R(lower)
    upper = R(upper)

    t = R(str(rng.random()))
    return lower + t * (upper - lower)


def sample_parameters(
    n_samples,
    beta_min=R("0.01"),
    beta_max=R("1"),
    eta_min=R("1e-8"),
    eta_max=R(e),
    c_min=R("4"),
    c_max=R("40"),
    seed=2026,
):
    """
    Generate parameter triples (x, eta, beta) given constraints.

    We sample c = beta*log(x), then set

        log(x) = c/beta,
        x = exp(c/beta).

    Sampling uses mixtures of uniform and log-uniform distributions:
      - beta: both ordinary-scale and small-beta coverage;
      - eta: log-uniform, since eta may vary over many orders of magnitude;
      - c: half concentrated near the threshold c=4, half spread across
        the full requested range.
    """
    beta_min = R(beta_min)
    beta_max = R(beta_max)
    eta_min = R(eta_min)
    eta_max = R(eta_max)
    c_min = R(c_min)
    c_max = R(c_max)

    if beta_min <= 0 or beta_max > 1 or beta_min >= beta_max:
        raise ValueError("Require 0 < beta_min < beta_max <= 1")

    if eta_min <= 0 or eta_max > R(e) or eta_min >= eta_max:
        raise ValueError("Require 0 < eta_min < eta_max <= e")

    if c_min < 4 or c_min >= c_max:
        raise ValueError("Require 4 <= c_min < c_max")

    rng = random.Random(seed)
    parameters = []

    for _ in range(n_samples):
        # Mixture sampling gives coverage both near beta_min and
        # throughout the full beta interval.
        if rng.random() < 0.5:
            beta = _log_uniform(rng, beta_min, beta_max)
        else:
            beta = _uniform(rng, beta_min, beta_max)

        # eta can naturally vary over many orders of magnitude.
        eta = _log_uniform(rng, eta_min, eta_max)

        # Concentrate half the samples near the hypothesis boundary c=4.
        if rng.random() < 0.5:
            near_upper = min(c_max, c_min + R(6))
            c = _uniform(rng, c_min, near_upper)
        else:
            c = _log_uniform(rng, c_min, c_max)

        L = c / beta
        x = exp(L)

        parameters.append({
            "x": x,
            "eta": eta,
            "beta": beta,
            "L": L,
            "c": c,
        })

    return parameters



"""
DELIBERATE BOUNDARY CASES
    Added manually to guarantee that these are hit in testing
"""

def boundary_parameters(
    beta_values=None,
    eta_values=None,
    c_values=None,
):
    """
    Generate a deterministic grid of important boundary and scale cases.
    """
    if beta_values is None:
        beta_values = [
            R("0.01"),
            R("0.02"),
            R("0.05"),
            R("0.1"),
            R("0.25"),
            R("0.5"),
            R("0.75"),
            R("1"),
        ]

    if eta_values is None:
        eta_values = [
            R("1e-8"),
            R("1e-6"),
            R("1e-4"),
            R("1e-2"),
            R("0.1"),
            R("0.5"),
            R("1"),
            R(e),
        ]

    if c_values is None:
        c_values = [
            R("4"),
            R("4.01"),
            R("4.1"),
            R("4.5"),
            R("5"),
            R("6"),
            R("8"),
            R("10"),
            R("20"),
            R("40"),
        ]

    parameters = []

    for beta in beta_values:
        beta = _validate_beta(beta)

        for eta in eta_values:
            eta = _validate_eta(eta)

            for c in c_values:
                c = R(c)
                L = c / beta
                x = exp(L)

                parameters.append({
                    "x": x,
                    "eta": eta,
                    "beta": beta,
                    "L": L,
                    "c": c,
                })

    return parameters




"""
SAMPLE EVALUATION
"""

def evaluate_point(x, eta, beta):
    """
    Evaluate the integral and upper bound at one parameter triple.
    """
    x = _validate_x(x, beta)
    eta = _validate_eta(eta)
    beta = _validate_beta(beta)

    integral, integration_error = actual_integral(x, eta, beta)
    bound = upper_bound(x, eta, beta)

    slack = bound - integral
    factor = bound / integral
    relative_slack = slack / integral
    efficiency = integral / bound

    L = log(x)
    c = beta * L

    return {
        "beta": beta,
        "eta": eta,
        "c": c,
        "L": L,
        "log10_x": L / log(R(10)),
        "integral": integral,
        "integration_error": integration_error,
        "bound": bound,
        "slack": slack,
        "factor": factor,
        "relative_slack": relative_slack,
        "efficiency": efficiency,
        "valid_bound": bound >= integral,
    }


def evaluate_samples(parameters, progress_every=100):
    """
    Evaluate all supplied parameter triples.
    """
    results = []
    total = len(parameters)

    for index, params in enumerate(parameters, start=1):
        result = evaluate_point(
            params["x"],
            params["eta"],
            params["beta"],
        )
        results.append(result)

        if progress_every and index % progress_every == 0:
            print(f"Evaluated {index}/{total} samples")

    return results



"""
COMPUTE STATISTICS
"""

def _quantile(values, q):
    """
    Compute a linearly interpolated quantile for 0 <= q <= 1.
    """
    if not values:
        raise ValueError("Cannot compute quantile of empty data")

    if q < 0 or q > 1:
        raise ValueError("q must satisfy 0 <= q <= 1")

    values = sorted(R(value) for value in values)

    if len(values) == 1:
        return values[0]

    position = R(q) * R(len(values) - 1)
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
    """
    Compute aggregate statistics for the upper-bound factor B/I.
    """
    if not results:
        raise ValueError("No results to summarize")

    factors = [row["factor"] for row in results]
    relative_slacks = [row["relative_slack"] for row in results]
    efficiencies = [row["efficiency"] for row in results]

    violations = [
        row for row in results
        if not row["valid_bound"]
    ]

    return {
        "number_samples": len(results),
        "number_violations": len(violations),
        "minimum_factor": min(factors),
        "median_factor": _quantile(factors, R("0.5")),
        "mean_factor": sum(factors) / len(factors),
        "p90_factor": _quantile(factors, R("0.9")),
        "p95_factor": _quantile(factors, R("0.95")),
        "p99_factor": _quantile(factors, R("0.99")),
        "maximum_factor": max(factors),
        "median_relative_slack": _quantile(
            relative_slacks, R("0.5")
        ),
        "median_efficiency": _quantile(
            efficiencies, R("0.5")
        ),
    }


"""
PRINT REPORTING
"""

def _print_result(row, title=None):
    if title is not None:
        print(title)
        print("-" * len(title))

    print(f"beta                 = {float(row['beta']):.8g}")
    print(f"eta                  = {float(row['eta']):.8g}")
    print(f"beta log(x)          = {float(row['c']):.8g}")
    print(f"log(x)               = {float(row['L']):.8g}")
    print(f"log10(x)             = {float(row['log10_x']):.8g}")
    print(f"actual integral      = {float(row['integral']):.10e}")
    print(f"upper bound          = {float(row['bound']):.10e}")
    print(f"absolute slack       = {float(row['slack']):.10e}")
    print(f"bound / integral     = {float(row['factor']):.8g}")
    print(f"relative slack       = {float(row['relative_slack']):.8g}")
    print(f"integral / bound     = {float(row['efficiency']):.8g}")
    print(f"integration error    = {float(row['integration_error']):.3e}")
    print()


def report_results(results, number_extreme_points=5):
    """
    Print aggregate statistics and the tightest/loosest points.
    """
    summary = summarize_results(results)

    print()
    print("=" * 72)
    print("GENERALIZED LEMMA C5 NUMERICAL REPORT")
    print("=" * 72)

    print(f"Number of samples:       {summary['number_samples']}")
    print(f"Bound violations:        {summary['number_violations']}")

    print()
    print("Upper-bound factor B/I")
    print("----------------------")
    print(f"Minimum:                 {float(summary['minimum_factor']):.8g}")
    print(f"Median:                  {float(summary['median_factor']):.8g}")
    print(f"Mean:                    {float(summary['mean_factor']):.8g}")
    print(f"90th percentile:         {float(summary['p90_factor']):.8g}")
    print(f"95th percentile:         {float(summary['p95_factor']):.8g}")
    print(f"99th percentile:         {float(summary['p99_factor']):.8g}")
    print(f"Maximum:                 {float(summary['maximum_factor']):.8g}")

    print()
    print(
        "Median relative slack:  "
        f"{float(summary['median_relative_slack']):.8g}"
    )
    print(
        "Median efficiency I/B: "
        f"{float(summary['median_efficiency']):.8g}"
    )

    sorted_by_factor = sorted(
        results,
        key=lambda row: row["factor"],
    )

    print()
    print("=" * 72)
    print("TIGHTEST SAMPLE POINTS")
    print("=" * 72)

    for index, row in enumerate(
        sorted_by_factor[:number_extreme_points],
        start=1,
    ):
        _print_result(row, f"Tight point {index}")

    print("=" * 72)
    print("LOOSEST SAMPLE POINTS")
    print("=" * 72)

    for index, row in enumerate(
        reversed(sorted_by_factor[-number_extreme_points:]),
        start=1,
    ):
        _print_result(row, f"Loose point {index}")



"""
CSV OUTPUT
"""

def write_results_csv(results, filename):
    """
    Save sample results to a CSV file.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "beta",
        "eta",
        "beta_log_x",
        "log_x",
        "log10_x",
        "integral",
        "integration_error",
        "upper_bound",
        "absolute_slack",
        "bound_over_integral",
        "relative_slack",
        "integral_over_bound",
        "valid_bound",
    ]

    with filename.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                "beta": str(row["beta"]),
                "eta": str(row["eta"]),
                "beta_log_x": str(row["c"]),
                "log_x": str(row["L"]),
                "log10_x": str(row["log10_x"]),
                "integral": str(row["integral"]),
                "integration_error": str(row["integration_error"]),
                "upper_bound": str(row["bound"]),
                "absolute_slack": str(row["slack"]),
                "bound_over_integral": str(row["factor"]),
                "relative_slack": str(row["relative_slack"]),
                "integral_over_bound": str(row["efficiency"]),
                "valid_bound": row["valid_bound"],
            })

    print(f"Wrote results to {filename}")




"""
GROUP-BY-PARAMETER REPORTING
"""

def report_group(
    results,
    predicate,
    label,
):
    group = [row for row in results if predicate(row)]

    if not group:
        return

    factors = [row["factor"] for row in group]

    print(
        f"{label:24s}"
        f" n={len(group):5d}"
        f" median={float(_quantile(factors, R('0.5'))):12.5g}"
        f" p90={float(_quantile(factors, R('0.9'))):12.5g}"
        f" max={float(max(factors)):12.5g}"
    )

def report_by_beta(results):
    print()
    print("FACTOR B/I GROUPED BY BETA")
    print("-" * 72)

    beta_bins = [
        (R("0"), R("0.05")),
        (R("0.05"), R("0.1")),
        (R("0.1"), R("0.25")),
        (R("0.25"), R("0.5")),
        (R("0.5"), R("0.75")),
        (R("0.75"), R("1.0000001")),
    ]

    for lower, upper in beta_bins:
        report_group(
            results,
            lambda row, a=lower, b=upper:
                a < row["beta"] <= b,
            f"{float(lower):.2g} < beta <= {float(upper):.2g}",
        )

def report_by_eta(results):
    print()
    print("FACTOR B/I GROUPED BY ETA")
    print("-" * 72)

    eta_bins = [
        (R("0"), R("1e-7")),
        (R("1e-7"), R("1e-5")),
        (R("1e-5"), R("1e-3")),
        (R("1e-3"), R("1e-1")),
        (R("1e-1"), R("1")),
        (R("1"), R(e) + R("1e-10")),
    ]

    for lower, upper in eta_bins:
        report_group(
            results,
            lambda row, a=lower, b=upper:
                a < row["eta"] <= b,
            f"{float(lower):.1e} < eta <= {float(upper):.1e}",
        )

# Group by c = beta log(x)
def report_by_c(results):
    print()
    print("FACTOR B/I GROUPED BY c = BETA LOG(X)")
    print("-" * 72)

    c_bins = [
        (R("4"), R("4.5")),
        (R("4.5"), R("6")),
        (R("6"), R("10")),
        (R("10"), R("20")),
        (R("20"), R("40.000001")),
    ]

    for lower, upper in c_bins:
        report_group(
            results,
            lambda row, a=lower, b=upper:
                a <= row["c"] < b,
            f"{float(lower):.1f} <= c < {float(upper):.1f}",
        )



"""
RUN TEST SUITE
"""

def run_test_suite(
    number_random_samples=10000,
    output_file="output/lemma_c5_samples.csv",
    seed=2026,
):
    print("Generating deterministic boundary samples...")
    parameters = boundary_parameters()

    print(f"Generating {number_random_samples} random samples...")
    parameters.extend(
        sample_parameters(
            n_samples=number_random_samples,
            beta_min=R("0.01"),
            beta_max=R("1"),
            eta_min=R("1e-8"),
            eta_max=R(e),
            c_min=R("4"),
            c_max=R("40"),
            seed=seed,
        )
    )

    print(f"Evaluating {len(parameters)} total samples...")
    results = evaluate_samples(parameters, progress_every=500)

    report_results(results, number_extreme_points=5)
    report_by_beta(results)
    report_by_eta(results)
    report_by_c(results)

    write_results_csv(results, output_file)

    return results


if __name__ == "__main__":
    for seed in [1234, 4321, 2143]:
        print()
        print("#" * 80)
        print(f"RUN WITH SEED {seed}")
        print("#" * 80)

        results = run_test_suite(
            number_random_samples=10000,
            output_file=f"output/lemma_c5_samples_seed_{seed}.csv",
            seed=seed,
        )


