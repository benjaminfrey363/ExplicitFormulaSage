
from pathlib import Path

import matplotlib.pyplot as plt

from sage.all import (
    ComplexField,
    RealField,
    zeta_zeros,
    log
)


RF = RealField(100)
CF = ComplexField(100)

I_NUM = CF.gen()
PI_NUM = RF.pi()





"""
Zero-loading
"""

def load_rh_zeros(n):
    """
    Load the first n positive zeta-zero ordinates from the
    Odlyzko database and return rho = 1/2 + i*gamma.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    ordinates = zeta_zeros()

    if n > len(ordinates):
        raise ValueError(
            f"Requested {n} zeros, but the database contains "
            f"{len(ordinates)}."
        )

    half = RF(1) / 2

    return [
        CF(half, RF(ordinates[index]))
        for index in range(n)
    ]




"""
Computation of CH Weight
"""

def theta_T_sigma(T, sigma):
    """
    Return the function theta_{T,sigma}(s).
    """
    T = RF(T)
    sigma = RF(sigma)
    if T <= 0:
        raise ValueError("T must be positive")

    def theta(s):
        s = CF(s)
        return 1 - (s - sigma) / (I_NUM * T)

    return theta


def c_T_sigma(T, sigma):
    """
    Compute the normalization constant c_{T,sigma}.
    """
    T = RF(T)
    theta = theta_T_sigma(T, sigma)
    theta_eval = theta(1 + I_NUM * T)

    return theta_eval * (PI_NUM * theta_eval).cot()


def omega_T_sigma(T, sigma):
    """
    Return the function omega^+_{T,sigma}(s).
    """
    c = c_T_sigma(T, sigma)
    theta = theta_T_sigma(T, sigma)

    def omega(s):
        theta_eval = theta(s)
        return -theta_eval * (PI_NUM * theta_eval).cot() + c

    return omega


def ch_weight_T_xi(T, xi):
    """
    Return the nonnegative CH weight

        |omega^+_{T,0}(s) + xi * theta_{T,1}(s) * i|

    as a function of s.
    """
    T = RF(T)
    xi = RF(xi)
    
    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    omega = omega_T_sigma(T, 0)
    theta = theta_T_sigma(T, 1)

    def ch_weight(s):
        s = CF(s)
        raw_weight = omega(s) + xi * theta(s) * I_NUM
        return RF(abs(raw_weight))

    return ch_weight




"""
Lemma 7.2 Pointwise bound
"""

def F(z):
    """
    Chirre--Helfgott auxiliary function

        F(z) = 1/pi - (1 - z) cot(pi(1 - z)).
    """
    z = CF(z)
    return 1 / PI_NUM - (1 - z) * (PI_NUM * (1 - z)).cot()


def ch_simplified_weight_T_xi(T, xi):
    """
    Return the simplified Lemma 7.2 weight

        |F(gamma/T) + xi(1 - gamma/T)i|

    as a function of s = beta + i*gamma.
    """
    T = RF(T)
    xi = RF(xi)

    if T <= 0:
        raise ValueError("T must be positive")
    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    def simplified_weight(s):
        s = CF(s)
        gamma = RF(s.imag())

        if not (0 < gamma <= T):
            raise ValueError("s must satisfy 0 < Im(s) <= T")

        raw_weight = (
            F(gamma / T)
            + xi * (1 - gamma / T) * I_NUM
        )

        return RF(abs(raw_weight))

    return simplified_weight


def ch_131_upper_bound_T_xi(T, xi):
    """
    Return the pointwise upper bound from equation (131),
    specialized to sigma = 0.

    For s = beta + i*gamma, this computes

        W_simp
        + beta*T/(pi*gamma^2)
        + (2 + 0.78*beta)/T.
    """
    T = RF(T)
    xi = RF(xi)

    simplified_weight = ch_simplified_weight_T_xi(T, xi)

    def upper_bound(s):
        s = CF(s)
        beta = RF(s.real())
        gamma = RF(s.imag())

        if not (0 < gamma <= T):
            raise ValueError("s must satisfy 0 < Im(s) <= T")
        if not (0 <= beta <= 1):
            raise ValueError("s must lie in the critical strip")

        reciprocal_square_error = beta * T / (PI_NUM * gamma**2)
        constant_order_error = (RF(2) + RF("0.78") * beta) / T

        return (
            simplified_weight(s)
            + reciprocal_square_error
            + constant_order_error
        )

    return upper_bound




"""
Pointwise weight computation
"""

def compute_weight_comparison(n, T, xi):
    """
    Compute exact, simplified, and equation (131) CH weights
    for the first n zeta zeros.

    Returns
    -------
    list[dict]
        One record for each zero.
    """
    T = RF(T)
    xi = RF(xi)

    if T <= 0:
        raise ValueError("T must be positive")
    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    zeros = load_rh_zeros(n)

    largest_gamma = zeros[-1].imag()
    if largest_gamma > T:
        raise ValueError(
            f"T must be at least the largest sampled ordinate. "
            f"For n={n}, require T >= {largest_gamma}."
        )

    exact_weight = ch_weight_T_xi(T, xi)
    simplified_weight = ch_simplified_weight_T_xi(T, xi)
    upper_bound_131 = ch_131_upper_bound_T_xi(T, xi)

    records = []

    for index, rho in enumerate(zeros, start=1):
        gamma = RF(rho.imag())

        exact = RF(exact_weight(rho))
        simplified = RF(simplified_weight(rho))
        upper = RF(upper_bound_131(rho))

        reciprocal_square_error = T / (2 * PI_NUM * gamma**2)
        constant_order_error = RF("2.39") / T

        records.append(
            {
                "index": index,
                "gamma": gamma,
                "exact": exact,
                "simplified": simplified,
                "upper_131": upper,
                "simp_minus_exact": simplified - exact,
                "upper_minus_exact": upper - exact,
                "upper_minus_simp": upper - simplified,
                "simp_relative_error": simplified / exact - 1,
                "upper_relative_gap": upper / exact - 1,
                "reciprocal_relative_gap": (
                    reciprocal_square_error / exact
                ),
                "constant_relative_gap": (
                    constant_order_error / exact
                ),
            }
        )

    return records




"""
PLOTTING FOR POINTWISE WEIGHTS
"""

def plot_weight_comparison(
    records,
    T,
    xi,
    output_path=None,
    show=True,
):
    """
    Plot exact, simplified, and equation (131) CH weights
    against the zero ordinate gamma.
    """
    if not records:
        raise ValueError("records must be nonempty")

    gammas = [float(record["gamma"]) for record in records]
    exact = [float(record["exact"]) for record in records]
    simplified = [
        float(record["simplified"])
        for record in records
    ]
    upper_131 = [
        float(record["upper_131"])
        for record in records
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        gammas,
        exact,
        label="Exact CH weight",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        simplified,
        label="Simplified weight",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        upper_131,
        label="Equation (131) upper bound",
        linewidth=1.5,
    )

    axis.set_xlabel(r"Zero ordinate $\gamma$")
    axis.set_ylabel("Pointwise weight")
    axis.set_title(
        "CH Lemma 7.2 pointwise weight comparison\n"
        f"First {len(records)} zeros, T={float(T):g}, "
        f"xi={float(xi):g}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)




def plot_weight_gaps(
    records,
    T,
    xi,
    output_path=None,
    show=True,
):
    """
    Plot the additive gaps between the exact, simplified,
    and equation (131) weights.
    """
    if not records:
        raise ValueError("records must be nonempty")

    gammas = [float(record["gamma"]) for record in records]

    simp_minus_exact = [
        float(record["simp_minus_exact"])
        for record in records
    ]
    upper_minus_exact = [
        float(record["upper_minus_exact"])
        for record in records
    ]
    upper_minus_simp = [
        float(record["upper_minus_simp"])
        for record in records
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        gammas,
        simp_minus_exact,
        label=r"$W_{\mathrm{simp}}-W_{\mathrm{exact}}$",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        upper_minus_exact,
        label=r"$W_{131}-W_{\mathrm{exact}}$",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        upper_minus_simp,
        label=r"$W_{131}-W_{\mathrm{simp}}$",
        linewidth=1.5,
    )

    axis.axhline(0, linewidth=1)

    axis.set_xlabel(r"Zero ordinate $\gamma$")
    axis.set_ylabel("Additive gap")
    axis.set_title(
        "CH Lemma 7.2 pointwise additive gaps\n"
        f"First {len(records)} zeros, T={float(T):.6g}, "
        f"xi={float(xi):.6g}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_weight_ratios(
    records,
    T,
    xi,
    output_path=None,
    show=True,
):
    """
    Plot the simplified and equation (131) weights relative
    to the exact CH weight.
    """
    if not records:
        raise ValueError("records must be nonempty")

    gammas = [float(record["gamma"]) for record in records]

    simp_over_exact = [
        float(record["simp_relative_error"])
        for record in records
    ]
    upper_over_exact = [
        float(record["upper_relative_gap"])
        for record in records
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        gammas,
        simp_over_exact,
        label=r"$(W_{\mathrm{simp}} - W_{\mathrm{exact}})/W_{\mathrm{exact}}$",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        upper_over_exact,
        label=r"$(W_{131} - W_{\mathrm{exact}})/W_{\mathrm{exact}}$",
        linewidth=1.5,
    )

    axis.axhline(
        0,
        linewidth=1,
        linestyle="--",
        label="Exact agreement",
    )

    axis.set_xlabel(r"Zero ordinate $\gamma$")
    axis.set_ylabel("Ratio to exact weight")
    axis.set_title(
        "CH Lemma 7.2 pointwise weight ratios\n"
        f"First {len(records)} zeros, "
        f"T={float(T):.6g}, xi={float(xi):.6g}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_relative_error_components(
    records,
    T,
    xi,
    output_path=None,
    show=True,
):
    """
    Decompose the relative gap in equation (131) into its
    reciprocal-square and constant-order components.
    """
    if not records:
        raise ValueError("records must be nonempty")

    gammas = [
        float(record["gamma"])
        for record in records
    ]
    total_gap = [
        float(record["upper_relative_gap"])
        for record in records
    ]
    reciprocal_gap = [
        float(record["reciprocal_relative_gap"])
        for record in records
    ]
    constant_gap = [
        float(record["constant_relative_gap"])
        for record in records
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        gammas,
        total_gap,
        label=r"Total relative gap",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        reciprocal_gap,
        label=r"$\frac{T}{2\pi\gamma^2W_{\rm exact}}$",
        linewidth=1.5,
    )
    axis.plot(
        gammas,
        constant_gap,
        label=r"$\frac{2.39}{T W_{\rm exact}}$",
        linewidth=1.5,
    )

    axis.set_xlabel(r"Zero ordinate $\gamma$")
    axis.set_ylabel("Relative contribution")
    axis.set_title(
        "Equation (131) relative error decomposition\n"
        f"First {len(records)} zeros, "
        f"T={float(T):.6g}, xi={float(xi):.6g}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(figure)





"""
PARTIAL SUM COMPUTATION
"""

def compute_normalized_partial_sums(
    zeros,
    T_values,
    xi,
):
    """
    Compute normalized CH partial sums for a fixed set of zeros
    over several truncation heights.

    Parameters
    ----------
    zeros : list
        Complex zeros rho = 1/2 + i*gamma.
    T_values : iterable
        Truncation heights. Each T must exceed every sampled ordinate.
    xi :
        Parameter in [-1, 1].

    Returns
    -------
    list[dict]
        One record for each truncation height.
    """
    if not zeros:
        raise ValueError("zeros must be nonempty")

    xi = RF(xi)

    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    largest_gamma = max(RF(rho.imag()) for rho in zeros)
    records = []

    for T_value in T_values:
        T = RF(T_value)

        if T <= largest_gamma:
            raise ValueError(
                f"Each T must exceed the largest sampled ordinate "
                f"{largest_gamma}; received T={T}."
            )

        exact_weight = ch_weight_T_xi(T, xi)
        simplified_weight = ch_simplified_weight_T_xi(T, xi)
        upper_bound_131 = ch_131_upper_bound_T_xi(T, xi)

        exact_sum = RF(0)
        simplified_sum = RF(0)
        upper_sum = RF(0)

        for rho in zeros:
            exact_sum += exact_weight(rho)
            simplified_sum += simplified_weight(rho)
            upper_sum += upper_bound_131(rho)

        normalization = 2 * PI_NUM / T

        normalized_exact = normalization * exact_sum
        normalized_simplified = normalization * simplified_sum
        normalized_upper = normalization * upper_sum

        records.append(
            {
                "T": T,
                "T_over_gamma_max": T / largest_gamma,
                "exact": normalized_exact,
                "simplified": normalized_simplified,
                "upper_131": normalized_upper,
                "simp_relative_error": (
                    normalized_simplified - normalized_exact
                ) / normalized_exact,
                "upper_relative_gap": (
                    normalized_upper - normalized_exact
                ) / normalized_exact,
            }
        )

    return records



"""
PARTIAL SUM PLOTTING
"""

def plot_normalized_partial_sums(
    records,
    n,
    xi,
    output_path=None,
    show=True,
):
    """
    Plot normalized exact, simplified, and equation (131)
    partial sums as T varies.
    """
    if not records:
        raise ValueError("records must be nonempty")

    x_values = [
        float(
            log(record["T_over_gamma_max"])
            / log(RF(10))
        )
        for record in records
    ]

    exact_values = [
        float(record["exact"])
        for record in records
    ]
    simplified_values = [
        float(record["simplified"])
        for record in records
    ]
    upper_values = [
        float(record["upper_131"])
        for record in records
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        x_values,
        exact_values,
        marker="o",
        label=r"$Z_N^{\mathrm{exact}}(T)$",
    )
    axis.plot(
        x_values,
        simplified_values,
        marker="o",
        label=r"$Z_N^{\mathrm{simp}}(T)$",
    )
    axis.plot(
        x_values,
        upper_values,
        marker="o",
        label=r"$Z_N^{131}(T)$",
    )

    axis.set_xlabel(
        r"$\log_{10}(T/\gamma_N)$"
    )
    axis.set_ylabel(
        r"Normalized partial sum $\frac{2\pi}{T}\sum W$"
    )
    axis.set_title(
        "Normalized CH partial sums as T varies\n"
        f"First {n} zeros, xi={float(xi):.6g}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_normalized_sum_relative_gaps(
    records,
    n,
    xi,
    output_path=None,
    show=True,
):
    """
    Plot relative errors of the simplified and equation (131)
    normalized partial sums.
    """
    if not records:
        raise ValueError("records must be nonempty")

    x_values = [
        float(log(record["T_over_gamma_max"], 10))
        for record in records
    ]

    simp_gaps = [
        float(record["simp_relative_error"])
        for record in records
    ]
    upper_gaps = [
        float(record["upper_relative_gap"])
        for record in records
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        x_values,
        simp_gaps,
        marker="o",
        label=(
            r"$(Z_N^{\mathrm{simp}}-"
            r"Z_N^{\mathrm{exact}})/Z_N^{\mathrm{exact}}$"
        ),
    )
    axis.plot(
        x_values,
        upper_gaps,
        marker="o",
        label=(
            r"$(Z_N^{131}-"
            r"Z_N^{\mathrm{exact}})/Z_N^{\mathrm{exact}}$"
        ),
    )

    axis.axhline(
        0,
        linewidth=1,
        linestyle="--",
    )

    axis.set_xlabel(
        r"$\log_{10}(T/\gamma_N)$"
    )
    axis.set_ylabel("Relative error")
    axis.set_title(
        "Relative gaps in normalized CH partial sums\n"
        f"First {n} zeros, xi={float(xi):.6g}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)



def reciprocal_square_limit(zeros):
    """
    Compute the predicted limiting normalized gap

        sum 1/gamma^2

    for zeros rho = 1/2 + i*gamma.
    """
    return sum(
        RF(1) / RF(rho.imag())**2
        for rho in zeros
    )





n = 1000
xi = RF(1)

zeros = load_rh_zeros(n)
gamma_max = RF(zeros[-1].imag())

scale_factors = [
    RF("1.01"),
    RF("1.1"),
    RF(2),
    RF(10),
    RF(100),
    RF(10**4),
    RF(10**8),
]

T_values = [
    factor * gamma_max
    for factor in scale_factors
]

records = compute_normalized_partial_sums(
    zeros=zeros,
    T_values=T_values,
    xi=xi,
)

print(
    "\n"
    "NORMALIZED PARTIAL SUMS\n"
    "-----------------------"
)

for record in records:
    print(
        f"T/gamma_max={float(record['T_over_gamma_max']):>12.5g}  "
        f"exact={float(record['exact']):>14.8g}  "
        f"simp={float(record['simplified']):>14.8g}  "
        f"upper={float(record['upper_131']):>14.8g}  "
        f"upper gap={float(record['upper_relative_gap']):>12.5g}"
    )


predicted_gap = reciprocal_square_limit(zeros)

last_record = records[-1]
observed_gap = (
    last_record["upper_131"]
    - last_record["exact"]
)

print("\nASYMPTOTIC GAP CHECK")
print("--------------------")
print(f"Observed gap at largest T: {observed_gap}")
print(f"Predicted sum 1/gamma^2:   {predicted_gap}")
print(f"Difference:                {observed_gap - predicted_gap}")

predicted_relative_gap = (
    predicted_gap / last_record["exact"]
)

print(
    f"Observed relative gap:  "
    f"{last_record['upper_relative_gap']}"
)
print(
    f"Predicted relative gap: "
    f"{predicted_relative_gap}"
)


"""
plot_normalized_partial_sums(
    records=records,
    n=n,
    xi=xi,
    output_path="output/ch_normalized_partial_sums.png",
    show=True,
)


plot_normalized_sum_relative_gaps(
    records=records,
    n=n,
    xi=xi,
    output_path="output/ch_normalized_partial_sum_gaps.png",
    show=True,
)
"""




"""
MAIN (DECOMP PLOTTING)
"""

"""
tolerance = RF("1e-20")

if __name__ == "__main__":
    n = 10000
    xi = RF(1)

    zeros = load_rh_zeros(n)
    largest_gamma = zeros[-1].imag()

    # Give some room between the final sampled zero and T.
    T = RF("1.1") * largest_gamma

    print(f"Loaded {n} zeros")
    print(f"Largest ordinate: {largest_gamma}")
    print(f"Using T: {T}")

    records = compute_weight_comparison(
        n=n,
        T=T,
        xi=xi,
    )

    for record in records:
        assert record["upper_minus_exact"] >= -tolerance

    plot_relative_error_components(
        records=records,
        T=T,
        xi=xi,
        output_path="output/ch_weight_ratios.png",
        show=True,
    )
"""





