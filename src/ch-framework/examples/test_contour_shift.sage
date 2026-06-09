"""
test_contour_shift.sage

Run from the repo root with:

    sage src/ch-framework/examples/test_contour_shift.sage
"""

load("src/ch-framework/weighted_perron.sage")
load("src/ch-framework/chirre_weights.sage")
load("src/ch-framework/contour_shift.sage")


def test_basic_zeta_contour_shift():
    x = 25.7
    c = 2
    alpha = 0.5
    T = 20

    result = contour_shift_zeta_perron(
        x=x,
        c=c,
        alpha=alpha,
        T=T,
    )

    print_contour_shift_result(result)


def test_chirre_regularized_zeta_shift_plus():
    x = 25.7
    sigma = 0
    T = 20
    alpha = 0.5
    side = "plus"

    result = contour_shift_chirre_regularized_zeta(
        x=x,
        sigma=sigma,
        T=T,
        alpha=alpha,
        side=side,
    )

    print_chirre_regularized_contour_shift_result(result)


def test_chirre_regularized_zeta_shift_minus():
    x = 25.7
    sigma = 0
    T = 20
    alpha = 0.5
    side = "minus"

    result = contour_shift_chirre_regularized_zeta(
        x=x,
        sigma=sigma,
        T=T,
        alpha=alpha,
        side=side,
    )

    print_chirre_regularized_contour_shift_result(result)


test_basic_zeta_contour_shift()
test_chirre_regularized_zeta_shift_plus()
test_chirre_regularized_zeta_shift_minus()