"""
test_smoothed_bounds.sage

Test the sum-side Chirre--Helfgott smoothing bounds.

Run from the repo root with:

    sage ch-framework/examples/test_smoothed_bounds.sage
"""

load("src/ch-framework/weighted_perron.sage")
load("src/ch-framework/chirre_weights.sage")
load("src/ch-framework/smoothed_bounds.sage")


# ============================================================
# Test coefficient functions
# ============================================================

def a_zeta(n):
    """
    Coefficients of zeta(s): a_n = 1.
    """
    return 1


def a_supported_small(n):
    """
    A small finitely supported nonnegative test sequence.

    This is useful because there is no truncation ambiguity.
    """
    values = {
        1: 2,
        2: 1,
        4: 3,
        7: 1,
        10: 5,
        15: 2,
        25: 4,
    }

    return values.get(n, 0)


# ============================================================
# Tests
# ============================================================

def test_finitely_supported_sigma_less_than_one():
    """
    Test sigma < 1 with finite support.
    """
    x = 12.5
    sigma = 0
    T = 20
    N_max = 50

    result = compare_sum_side_bounds(
        a=a_supported_small,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_sum_side_comparison(result)


def test_finitely_supported_sigma_greater_than_one():
    """
    Test sigma > 1 with finite support.
    """
    x = 12.5
    sigma = 2
    T = 20
    N_max = 50

    result = compare_sum_side_bounds(
        a=a_supported_small,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_sum_side_comparison(result)


def test_zeta_coefficients_sigma_less_than_one():
    """
    Test with a_n = 1 and sigma < 1.

    For sigma < 1, S_sigma(x) is finite, so truncation is harmless as long
    as N_max >= floor(x).
    """
    x = 25.7
    sigma = 0
    T = 20
    N_max = 200

    result = compare_sum_side_bounds(
        a=a_zeta,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_sum_side_comparison(result)


def test_zeta_coefficients_sigma_greater_than_one():
    """
    Test with a_n = 1 and sigma > 1.

    For sigma > 1, S_sigma(x) is an infinite tail, so this test is only
    truncated up to N_max.
    """
    x = 25.7
    sigma = 2
    T = 20
    N_max = 500

    result = compare_sum_side_bounds(
        a=a_zeta,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print_sum_side_comparison(result)


# ============================================================
# Run tests
# ============================================================

test_finitely_supported_sigma_less_than_one()
test_finitely_supported_sigma_greater_than_one()
test_zeta_coefficients_sigma_less_than_one()
test_zeta_coefficients_sigma_greater_than_one()