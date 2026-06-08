"""
PerronsFormula.sage

Exploratory Sage script studying Perron's formula, working toward
implementing Chirre's Perron-type formula.

Classical Perron's formula says roughly that if

    F(s) = sum_{n >= 1} a_n / n^s,

then for c > abscissa of convergence,

    sum_{n <= x} a_n
    approx
    (1 / 2 pi i) int_{c - iT}^{c + iT} F(s) x^s / s ds.

After parametrizing s = c + it, this becomes

    (1 / 2 pi) int_{-T}^{T} F(c + it) x^(c+it)/(c+it) dt.
"""


def perron_integral(F, x, c=2, T=50):
    """
    Numerically compute the truncated Perron integral

        (1 / 2 pi i) int_{c-iT}^{c+iT} F(s) x^s / s ds.

    Parameters
    ----------
    F : function
        Function of a complex variable s.
    x : real
        Cutoff parameter.
    c : real
        Real part of the vertical line of integration.
    T : real
        Height cutoff.

    Returns
    -------
    complex
        Numerical approximation to the Perron integral.
    """
    def integrand(t):
        s = c + I*t
        return F(s) * x**s / s

    real_part = numerical_integral(lambda t: real(integrand(t)), -T, T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), -T, T)[0]

    return (real_part + I*imag_part) / (2*pi)


def F_zeta(s):
    """
    Dirichlet series F(s) = zeta(s).

    The coefficients are a_n = 1, so Perron's formula should recover
    approximately floor(x).
    """
    return zeta(s)


def test_zeta_perron(x=25.7, c=2, T_values=[5, 10, 20, 50, 100]):
    """
    Test Perron's formula with F(s) = zeta(s).

    Since zeta(s) = sum 1/n^s, the target value is floor(x).
    """
    exact = floor(x)

    print(f"Testing Perron's formula for F(s) = zeta(s)")
    print(f"x = {x}")
    print(f"Exact floor(x) = {exact}")
    print()

    for T in T_values:
        approx = perron_integral(F_zeta, x=x, c=c, T=T)

        print(f"T = {T}")
        print(f"  approximation = {N(approx)}")
        print(f"  real part     = {N(real(approx))}")
        print(f"  imaginary     = {N(imag(approx))}")
        print(f"  error         = {N(real(approx) - exact)}")
        print()


# Run demo if this file is executed directly by Sage.
test_zeta_perron()