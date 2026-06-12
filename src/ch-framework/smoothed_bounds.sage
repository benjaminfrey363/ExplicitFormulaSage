"""
smoothed_bounds.sage

Stage 1 (kinda) of the Chirre--Helfgott framework.

This file implements the sum-side version of the smoothing argument.

The main sharp sum is

    S_sigma(x) = sum_{n <= x} a_n / n^sigma      if sigma < 1,

and

    S_sigma(x) = sum_{n >= x} a_n / n^sigma      if sigma > 1.

Chirre--Helfgott encode this sharp cutoff using the truncated exponential

    I_lambda(y),

where

    lambda = 2*pi*(sigma - 1)/T
    y = T/(2*pi) * log(n/x).

The identity is

    S_sigma(x)
    =
    x^(-sigma) sum_n a_n * x/n * I_lambda(T/(2*pi) log(n/x)).

Since a_n >= 0, if

    phi_hat^-(y) <= I_lambda(y) <= phi_hat^+(y),

then the corresponding smoothed sums bound S_sigma(x) from below and above.

This file only implements the sum-side bounds. It does not yet implement
the integral-side formula of Proposition 2.4.
"""


# ============================================================
# Validation helpers
# ============================================================

def validate_sigma_not_one(sigma):
    """
    Chirre--Helfgott define S_sigma separately for sigma != 1.
    """
    if sigma == 1:
        raise ValueError("sigma must not equal 1 in this stage")


def validate_side(side):
    """
    Check that side is either 'plus' or 'minus'.
    """
    if side not in {"plus", "minus"}:
        raise ValueError("side must be either 'plus' or 'minus'")


# ============================================================
# Sharp sum S_sigma(x)
# ============================================================

def sharp_sum_S_sigma(a, x, sigma, N_max):
    """
    Compute a finite version of the sharp sum S_sigma(x).

    If sigma < 1:

        S_sigma(x) = sum_{n <= x} a_n / n^sigma.

    If sigma > 1:

        S_sigma(x) = sum_{n >= x} a_n / n^sigma.

    Since we are computing numerically, the sigma > 1 case is truncated
    at N_max:

        sum_{ceil(x) <= n <= N_max} a_n / n^sigma.

    Parameters
    ----------
    a : function
        Coefficient function n |-> a_n.
    x : real
        Cutoff parameter.
    sigma : real
        Exponent parameter, sigma != 1.
    N_max : int
        Maximum n used in the computation.

    Returns
    -------
    real or complex
        Truncated sharp sum.
    """
    validate_sigma_not_one(sigma)

    if x <= 0:
        raise ValueError("x must be positive")

    if N_max < 1:
        raise ValueError("N_max must be at least 1")

    total = 0

    if sigma < 1:
        upper = min(floor(x), N_max)

        for n in range(1, upper + 1):
            total += a(n) / (n**sigma)

    else:
        lower = max(ceil(x), 1)

        for n in range(lower, N_max + 1):
            total += a(n) / (n**sigma)

    return total


# ============================================================
# Sharp sum encoded using I_lambda
# ============================================================

def sharp_sum_via_I_lambda(a, x, sigma, T, N_max):
    """
    Compute the I_lambda-encoded version of S_sigma(x):

        x^(-sigma) sum_n a_n * x/n *
            I_lambda(T/(2*pi) log(n/x)).

    This should agree with sharp_sum_S_sigma(a, x, sigma, N_max),
    up to the same truncation convention.

    Parameters
    ----------
    a : function
        Coefficient function n |-> a_n.
    x : real
        Cutoff parameter.
    sigma : real
        Exponent parameter, sigma != 1.
    T : real
        Height parameter.
    N_max : int
        Maximum n used in the computation.

    Returns
    -------
    real or complex
        I_lambda-encoded sharp sum.
    """
    validate_sigma_not_one(sigma)

    if T <= 0:
        raise ValueError("T must be positive")

    lam = chirre_lambda(sigma, T)

    total = 0

    for n in range(1, N_max + 1):
        y = T/(2*pi) * log(n/x)
        total += a(n) * (x/n) * truncated_exponential_I_lambda(y, lam)

    return x**(-sigma) * total


# ============================================================
# Chirre smoothed sum bounds
# ============================================================

def chirre_smoothed_sum(a, x, sigma, T, side, N_max):
    """
    Compute the smoothed sum using Chirre--Helfgott's majorant/minorant:

        x^(-sigma) sum_n a_n * x/n *
            hat{phi_lambda^side}(T/(2*pi) log(n/x)).

    side = 'minus' should give a lower bound.

    side = 'plus' should give an upper bound.

    Parameters
    ----------
    a : function
        Coefficient function n |-> a_n.
    x : real
        Cutoff parameter.
    sigma : real
        Exponent parameter, sigma != 1.
    T : real
        Height parameter.
    side : str
        Either 'minus' or 'plus'.
    N_max : int
        Maximum n used in the computation.

    Returns
    -------
    real or complex
        Smoothed sum.
    """
    validate_sigma_not_one(sigma)
    validate_side(side)

    if T <= 0:
        raise ValueError("T must be positive")

    lam = chirre_lambda(sigma, T)
    phi_hat = make_chirre_phi_hat_numeric(lam, side)

    total = 0

    for n in range(1, N_max + 1):
        y = T/(2*pi) * log(n/x)
        total += a(n) * (x/n) * phi_hat(y)

    return x**(-sigma) * total


def chirre_lower_smoothed_sum(a, x, sigma, T, N_max):
    """
    Convenience wrapper for the lower smoothed sum.
    """
    return chirre_smoothed_sum(
        a=a,
        x=x,
        sigma=sigma,
        T=T,
        side="minus",
        N_max=N_max,
    )


def chirre_upper_smoothed_sum(a, x, sigma, T, N_max):
    """
    Convenience wrapper for the upper smoothed sum.
    """
    return chirre_smoothed_sum(
        a=a,
        x=x,
        sigma=sigma,
        T=T,
        side="plus",
        N_max=N_max,
    )


# ============================================================
# Comparison helpers
# ============================================================

def compare_sum_side_bounds(a, x, sigma, T, N_max):
    """
    Compare the lower smoothed sum, sharp sum, I_lambda sharp sum,
    and upper smoothed sum.

    Returns
    -------
    dict
        Dictionary containing all values and inequality violations.
    """
    lower = real(chirre_lower_smoothed_sum(a, x, sigma, T, N_max))
    sharp = sharp_sum_S_sigma(a, x, sigma, N_max)
    sharp_I = real(sharp_sum_via_I_lambda(a, x, sigma, T, N_max))
    upper = real(chirre_upper_smoothed_sum(a, x, sigma, T, N_max))

    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "N_max": N_max,
        "lower_smoothed": lower,
        "sharp_sum": sharp,
        "sharp_sum_via_I_lambda": sharp_I,
        "upper_smoothed": upper,
        "sharp_minus_I_encoded": sharp - sharp_I,
        "lower_violation": max(0, lower - sharp),
        "upper_violation": max(0, sharp - upper),
    }


def print_sum_side_comparison(result):
    """
    Pretty-print the output of compare_sum_side_bounds.
    """
    print("=" * 70)
    print("Chirre sum-side smoothing comparison")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"N_max = {result['N_max']}")
    print()
    print(f"lower smoothed sum       = {N(result['lower_smoothed'])}")
    print(f"sharp sum                = {N(result['sharp_sum'])}")
    print(f"sharp sum via I_lambda   = {N(result['sharp_sum_via_I_lambda'])}")
    print(f"upper smoothed sum       = {N(result['upper_smoothed'])}")
    print()
    print(f"sharp - I_encoded        = {N(result['sharp_minus_I_encoded'])}")
    print(f"lower violation          = {N(result['lower_violation'])}")
    print(f"upper violation          = {N(result['upper_violation'])}")
    print()
    