"""
test_explicit_formula_chirre.sage

Run from the repo root with:

    sage src/ch-framework/examples/test_explicit_formula_chirre.sage
"""

load("src/ch-framework/weighted_perron.sage")
load("src/ch-framework/chirre_weights.sage")
load("src/ch-framework/smoothed_bounds.sage")
load("src/ch-framework/integral_bounds.sage")
load("src/ch-framework/contour_shift.sage")
load("src/ch-framework/explicit_formula_chirre.sage")


def test_chirre_explicit_formula_plus():
    result = chirre_explicit_formula_diagnostic(
        x=25.7,
        sigma=0,
        T=45,
        alpha=0.25,
        side="plus",
        N_max=300,
    )

    print_chirre_explicit_formula_diagnostic(result)


def test_chirre_explicit_formula_minus():
    result = chirre_explicit_formula_diagnostic(
        x=25.7,
        sigma=0,
        T=45,
        alpha=0.25,
        side="minus",
        N_max=300,
    )

    print_chirre_explicit_formula_diagnostic(result)


test_chirre_explicit_formula_plus()
test_chirre_explicit_formula_minus()