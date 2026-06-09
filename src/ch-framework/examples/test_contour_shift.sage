"""
test_contour_shift.sage

Run from the repo root with:

    sage src/ch-framework/examples/test_contour_shift.sage
"""

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


def test_larger_T():
    x = 25.7
    c = 2
    alpha = 0.5
    T = 50

    result = contour_shift_zeta_perron(
        x=x,
        c=c,
        alpha=alpha,
        T=T,
    )

    print_contour_shift_result(result)


test_basic_zeta_contour_shift()
test_larger_T()