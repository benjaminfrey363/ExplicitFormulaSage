"""
test_integral_bounds.sage

Test the integral-side Chirre--Helfgott bounds.

Run from the repo root with:

    sage ch-framework/examples/test_integral_bounds.sage
"""

load("src/ch-framework/weighted_perron.sage")
load("src/ch-framework/chirre_weights.sage")
load("src/ch-framework/smoothed_bounds.sage")
load("src/ch-framework/integral_bounds.sage")


# ============================================================
# Test case: A(s) = zeta(s), a_n = 1
# ============================================================

def a_zeta(n):
    """
    Coefficients of zeta(s): a_n = 1.
    """
    return 1


def zeta_regularized(s):
    """
    Regularized zeta function:

        zeta(s) - 1/(s - 1).

    This is regular at s = 1, with value Euler's constant.

    We handle s = 1 separately because numerical integration may sample
    u = 0, corresponding to s = 1.
    """
    if abs(s - 1) < 1e-10:
        return euler_gamma

    return zeta(s) - 1/(s - 1)


# ============================================================
# Tests
# ============================================================

def test_zeta_sigma_zero():
    """
    Test sigma = 0.

    Here

        S_0(x) = sum_{n <= x} 1 = floor(x).

    This is the cleanest first test because the sharp sum is finite.
    """
    x = 25.7
    sigma = 0
    T = 20
    N_max = 100

    result = compare_integral_bounds_to_sharp_sum(
        a=a_zeta,
        F_regular=zeta_regularized,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_integral_bound_comparison(result)


def test_zeta_sigma_half():
    """
    Test sigma = 1/2.

    Here

        S_{1/2}(x) = sum_{n <= x} 1/sqrt(n).

    Again this is finite because sigma < 1.
    """
    x = 25.7
    sigma = 1/2
    T = 20
    N_max = 100

    result = compare_integral_bounds_to_sharp_sum(
        a=a_zeta,
        F_regular=zeta_regularized,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_integral_bound_comparison(result)


def test_zeta_sigma_two_truncated_tail():
    """
    Test sigma = 2.

    Here

        S_2(x) = sum_{n >= x} 1/n^2.

    Our sharp sum is truncated at N_max, while the integral bound is
    designed for the infinite tail. So we should not expect as perfect
    a diagnostic as in the sigma < 1 tests.

    Still, the lower/upper bounds should be plausibly placed.
    """
    x = 25.7
    sigma = 2
    T = 20
    N_max = 10000

    result = compare_integral_bounds_to_sharp_sum(
        a=a_zeta,
        F_regular=zeta_regularized,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_integral_bound_comparison(result)


# ============================================================
# Run tests
# ============================================================

test_zeta_sigma_zero()
test_zeta_sigma_half()
test_zeta_sigma_two_truncated_tail()