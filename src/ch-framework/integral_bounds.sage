"""
integral_bounds.sage

Stage 2 of the Chirre--Helfgott framework.

This file implements the integral-side upper/lower bounds from the
Proposition 2.4 layer.

At this stage we assume:

    A(s) = sum a_n n^(-s)

has a simple pole at s = 1 with residue 1, and we work with the
regularized function

    F_regular(s) = A(s) - 1/(s - 1).

Then the smoothed upper/lower bounds can be written using the vertical
integral over the segment from 1 - iT to 1 + iT.

This is still before the full contour shift. No residues from nontrivial
zeros are being collected yet.
"""


# ============================================================
# Validation helpers
# ============================================================

def validate_sigma_not_one_integral(sigma):
    """
    The current Proposition 2.4 implementation assumes sigma != 1.
    """
    if sigma == 1:
        raise ValueError("sigma must not equal 1")


def validate_integral_side(side):
    """
    Check that side is either 'plus' or 'minus'.
    """
    if side not in {"plus", "minus"}:
        raise ValueError("side must be either 'plus' or 'minus'")


# ============================================================
# Pole correction and phi(0)
# ============================================================

def chirre_phi_zero(sigma, T, side):
    """
    Compute phi_lambda^side(0), where

        lambda = 2*pi*(sigma - 1)/T.

    For the Chirre--Helfgott weights this is also given explicitly by

        phi_lambda^plus(0)  = 1/2 coth(|lambda|/2) + 1/2,
        phi_lambda^minus(0) = 1/2 coth(|lambda|/2) - 1/2.

    We compute it directly from chirre_phi_lambda to keep the code
    consistent with chirre_weights.sage.
    """
    validate_sigma_not_one_integral(sigma)
    validate_integral_side(side)

    lam = chirre_lambda(sigma, T)
    return chirre_phi_lambda(0, lam, side)


def pole_main_term(sigma, T, x, side):
    """
    Compute the pole contribution appearing in the integral-side bounds:

        2*pi*x^(1-sigma)/T * phi_lambda^side(0).

    This assumes Res_{s=1} A(s) = 1.
    """
    phi0 = chirre_phi_zero(sigma, T, side)
    return (2*pi*x**(1 - sigma)/T) * phi0


def sharp_cutoff_tail_correction(sigma):
    """
    Compute the correction term

        1_{sigma < 1} / (sigma - 1).

    This is the same as

        -1_{sigma < 1} / (1 - sigma).

    It appears when converting the smoothed Perron formula into a bound
    for S_sigma(x).
    """
    validate_sigma_not_one_integral(sigma)

    if sigma < 1:
        return 1/(sigma - 1)

    return 0


# ============================================================
# Vertical integral term
# ============================================================

def regularized_vertical_integral(F_regular, x, sigma, T, side):
    """
    Compute the regularized vertical integral term

        x^(-sigma)/(iT) int_{1-iT}^{1+iT}
            phi_lambda^side((s-1)/(iT)) F_regular(s) x^s ds.

    Here

        F_regular(s) = A(s) - 1/(s - 1).

    Parametrize

        s = 1 + iT*u,    -1 <= u <= 1.

    Then ds = iT du, so this becomes

        x^(-sigma) int_{-1}^{1}
            phi_lambda^side(u) F_regular(1+iT*u) x^(1+iT*u) du.
    """
    validate_sigma_not_one_integral(sigma)
    validate_integral_side(side)

    lam = chirre_lambda(sigma, T)
    phi = make_chirre_phi(lam, side)

    def integrand(u):
        s = 1 + I*T*u
        return phi(u) * F_regular(s) * x**s

    real_part = numerical_integral(lambda u: real(integrand(u)), -1, 1)[0]
    imag_part = numerical_integral(lambda u: imag(integrand(u)), -1, 1)[0]

    return x**(-sigma) * (real_part + I*imag_part)


# ============================================================
# Integral-side lower/upper bounds
# ============================================================

def chirre_integral_bound(F_regular, x, sigma, T, side):
    """
    Compute the integral-side Chirre--Helfgott bound.

    side = 'minus' gives the lower bound.

    side = 'plus' gives the upper bound.

    The formula implemented is:

        2*pi*x^(1-sigma)/T * phi_lambda^side(0)
        + x^(-sigma)/(iT) int_{1-iT}^{1+iT}
            phi_lambda^side((s-1)/(iT))
            F_regular(s) x^s ds
        + 1_{sigma < 1}/(sigma - 1).

    This assumes Res_{s=1} A(s) = 1.
    """
    validate_sigma_not_one_integral(sigma)
    validate_integral_side(side)

    main = pole_main_term(sigma, T, x, side)
    integral = regularized_vertical_integral(
        F_regular=F_regular,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )
    correction = sharp_cutoff_tail_correction(sigma)

    return main + integral + correction


def chirre_integral_lower_bound(F_regular, x, sigma, T):
    """
    Lower integral-side bound.
    """
    return chirre_integral_bound(
        F_regular=F_regular,
        x=x,
        sigma=sigma,
        T=T,
        side="minus",
    )


def chirre_integral_upper_bound(F_regular, x, sigma, T):
    """
    Upper integral-side bound.
    """
    return chirre_integral_bound(
        F_regular=F_regular,
        x=x,
        sigma=sigma,
        T=T,
        side="plus",
    )


# ============================================================
# Comparison helper
# ============================================================

def compare_integral_bounds_to_sharp_sum(a, F_regular, x, sigma, T, N_max):
    """
    Compare integral-side bounds against the sharp sum S_sigma(x).

    This uses sharp_sum_S_sigma from smoothed_bounds.sage.

    For sigma < 1, the sharp sum is finite, so N_max only needs to be
    at least floor(x).

    For sigma > 1, the sharp sum is an infinite tail, and the comparison
    is only against the truncated tail up to N_max.
    """
    lower = real(chirre_integral_lower_bound(F_regular, x, sigma, T))
    sharp = sharp_sum_S_sigma(a, x, sigma, N_max)
    upper = real(chirre_integral_upper_bound(F_regular, x, sigma, T))

    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "N_max": N_max,
        "lower_integral_bound": lower,
        "sharp_sum": sharp,
        "upper_integral_bound": upper,
        "lower_violation": max(0, lower - sharp),
        "upper_violation": max(0, sharp - upper),
    }


def print_integral_bound_comparison(result):
    """
    Pretty-print output of compare_integral_bounds_to_sharp_sum.
    """
    print("=" * 70)
    print("Chirre integral-side bound comparison")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"N_max = {result['N_max']}")
    print()
    print(f"lower integral bound = {N(result['lower_integral_bound'])}")
    print(f"sharp sum            = {N(result['sharp_sum'])}")
    print(f"upper integral bound = {N(result['upper_integral_bound'])}")
    print()
    print(f"lower violation      = {N(result['lower_violation'])}")
    print(f"upper violation      = {N(result['upper_violation'])}")
    print()