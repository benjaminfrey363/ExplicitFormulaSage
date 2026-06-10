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


def test_chirre_pieces_regularized_zeta_plus():
    result = contour_shift_chirre_pieces_regularized_zeta(
        x=25.7,
        sigma=0,
        T=20,
        alpha=0.5,
        side="plus",
    )
    print_chirre_pieces_contour_shift_result(result)


def test_chirre_pieces_regularized_zeta_minus():
    result = contour_shift_chirre_pieces_regularized_zeta(
        x=25.7,
        sigma=0,
        T=20,
        alpha=0.5,
        side="minus",
    )
    print_chirre_pieces_contour_shift_result(result)



def test_chirre_toy_pole_upper_plus():
    """
    Toy pole in the upper half-plane.
    """
    rho = 0.75 + 3*I

    result = contour_shift_chirre_toy_pole(
        rho=rho,
        x=25.7,
        sigma=0,
        T=20,
        alpha=0.5,
        side="plus",
    )

    print_chirre_toy_pole_contour_shift_result(result)


def test_chirre_toy_pole_lower_plus():
    """
    Toy pole in the lower half-plane.

    This checks the sign convention for the lower star rectangle.
    """
    rho = 0.75 - 3*I

    result = contour_shift_chirre_toy_pole(
        rho=rho,
        x=25.7,
        sigma=0,
        T=20,
        alpha=0.5,
        side="plus",
    )

    print_chirre_toy_pole_contour_shift_result(result)


def test_chirre_first_zeta_zero_pair_plus():
    result = contour_shift_chirre_zeta_zero_pair(
        x=25.7,
        sigma=0,
        T=20,
        alpha=0.25,
        side="plus",
    )

    print_chirre_zeta_zero_pair_result(result)


def test_chirre_first_zeta_zero_pair_minus():
    result = contour_shift_chirre_zeta_zero_pair(
        x=25.7,
        sigma=0,
        T=20,
        alpha=0.25,
        side="minus",
    )

    print_chirre_zeta_zero_pair_result(result)



test_chirre_first_zeta_zero_pair_plus()
test_chirre_first_zeta_zero_pair_minus()
