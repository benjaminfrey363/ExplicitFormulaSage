"""
test_chirre_weights.sage

Diagnostic tests for chirre_weights.sage.

Run from the repo root with:

    sage ch-framework/examples/test_chirre_weights.sage
"""

load("src/ch-framework/weighted_perron.sage")
load("src/ch-framework/chirre_weights.sage")


def print_chirre_weight_table():
    """
    Print a few values comparing the minorant, target, and majorant.
    """
    sigma = 2
    T = 20
    lam = chirre_lambda(sigma, T)

    y_values = [-3, -1, -0.25, 0, 0.25, 1, 3]

    print("=" * 70)
    print("Chirre weight sanity check")
    print("=" * 70)
    print(f"sigma = {sigma}")
    print(f"T = {T}")
    print(f"lambda = {N(lam)}")
    print()
    print("y        minorant_hat        I_lambda          majorant_hat")
    print("-" * 70)

    for y in y_values:
        row = chirre_weight_values_at_y(y, lam)

        print(
            f"{y:<8} "
            f"{N(row['minorant_hat']):<18} "
            f"{N(row['I_lambda']):<18} "
            f"{N(row['majorant_hat'])}"
        )

    print()


def print_chirre_inequality_grid_test():
    """
    Check the majorant/minorant inequalities on a small grid.

    This uses numerical Fourier transforms, so keep the grid small.
    """
    sigma = 2
    T = 20
    lam = chirre_lambda(sigma, T)

    y_values = [-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3]

    result = chirre_weight_inequality_grid(lam, y_values)

    print("=" * 70)
    print("Chirre weight inequality grid test")
    print("=" * 70)
    print(f"lambda = {N(result['lambda'])}")
    print()
    print("Worst lower violation:", N(result["worst_lower_violation"]))
    print("Worst upper violation:", N(result["worst_upper_violation"]))
    print()


def print_lambda_sign_test():
    """
    Test both lambda > 0 and lambda < 0.

    The lambda < 0 case matters because Chirre--Helfgott treat sigma < 1
    and sigma > 1 using the same truncated exponential notation.
    """
    T = 20
    sigma_values = [0, 2]

    print("=" * 70)
    print("Lambda sign test")
    print("=" * 70)

    for sigma in sigma_values:
        lam = chirre_lambda(sigma, T)
        I_lam = make_I_lambda(lam)

        print(f"sigma = {sigma}")
        print(f"lambda = {N(lam)}")
        print("y        I_lambda(y)")
        print("-" * 40)

        for y in [-2, -1, 0, 1, 2]:
            print(f"{y:<8} {N(I_lam(y))}")

        print()


print_chirre_weight_table()
print_chirre_inequality_grid_test()
print_lambda_sign_test()
