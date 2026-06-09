"""
PerronsFormula.sage

Exploratory Sage script studying Perron's formula, working toward
implementing Chirre's Perron-type formula.

The classical Perron formula has the shape

    sum_{n <= x} a_n
    approx
    (1 / 2 pi i) int_{c-iT}^{c+iT} F(s) x^s / s ds,

where

    F(s) = sum_{n >= 1} a_n / n^s.

More generally, Perron-type formulae replace the classical kernel 1/s
with another kernel K(s), giving an integral of the form

    (1 / 2 pi i) int_{c-iT}^{c+iT} F(s) x^s K(s) ds.

Classical Perron is recovered by choosing

    K(s) = 1/s.
"""


# Load chebyshev prime-counting functions
load("src/primecounting/chebyshev.py")


# ============================================================
# General Perron-type integral
# ============================================================

def perron_type_integral(F, K, x, c=2, T=50):
    """
    Numerically compute the truncated Perron-type integral

        (1 / 2 pi i) int_{c-iT}^{c+iT} F(s) x^s K(s) ds.

    Parametrizing s = c + it gives

        (1 / 2 pi) int_{-T}^{T} F(c+it) x^(c+it) K(c+it) dt.

    Parameters
    ----------
    F : function
        Function of a complex variable s. Usually a Dirichlet series.
    K : function
        Kernel function of a complex variable s.
    x : real
        Cutoff parameter.
    c : real
        Real part of the vertical line of integration.
    T : real
        Height cutoff.

    Returns
    -------
    complex
        Numerical approximation to the Perron-type integral.
    """
    def integrand(t):
        s = c + I*t
        return F(s) * x**s * K(s)

    real_part = numerical_integral(lambda t: real(integrand(t)), -T, T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), -T, T)[0]

    return (real_part + I*imag_part) / (2*pi)


# ============================================================
# Classical Perron kernel
# ============================================================

def K_classical(s):
    """
    Classical Perron kernel.

    This corresponds to the sharp cutoff sum

        sum_{n <= x} a_n.
    """
    return 1/s


def perron_integral(F, x, c=2, T=50):
    """
    Classical Perron integral.

    This is the special case of perron_type_integral with K(s) = 1/s.
    """
    return perron_type_integral(F, K_classical, x=x, c=c, T=T)


# ============================================================
# Example 1: F(s) = zeta(s)
# ============================================================

def F_zeta(s):
    """
    Dirichlet series F(s) = zeta(s).

    Since

        zeta(s) = sum_{n >= 1} 1/n^s,

    the coefficients are a_n = 1. Therefore classical Perron's formula
    should recover approximately

        sum_{n <= x} 1 = floor(x).
    """
    return zeta(s)


def exact_sum_for_zeta(x):
    """
    Exact target value for F(s) = zeta(s).

    Since a_n = 1, the exact sum is floor(x).
    """
    return floor(x)


def test_zeta_perron(x=25.7, c=2, T_values=[5, 10, 20, 50, 100]):
    """
    Test classical Perron's formula with F(s) = zeta(s).
    """
    exact = exact_sum_for_zeta(x)

    print("=" * 70)
    print("Classical Perron test: F(s) = zeta(s)")
    print("=" * 70)
    print(f"x = {x}")
    print(f"c = {c}")
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


# ============================================================
# Example 2: Partial sums away from integers
# ============================================================

def test_zeta_perron_at_multiple_x(c=2, T=50):
    """
    Test Perron's formula for several x-values.

    Perron's formula has jump behavior near integers. It tends to behave
    more cleanly when x is not an integer.
    """
    x_values = [5.5, 10.3, 25.7, 50.2, 100.8]

    print("=" * 70)
    print("Classical Perron test for several x-values")
    print("=" * 70)
    print(f"c = {c}")
    print(f"T = {T}")
    print()

    for x in x_values:
        exact = exact_sum_for_zeta(x)
        approx = perron_integral(F_zeta, x=x, c=c, T=T)

        print(f"x = {x}")
        print(f"  exact floor(x) = {exact}")
        print(f"  approximation  = {N(real(approx))}")
        print(f"  error          = {N(real(approx) - exact)}")
        print(f"  imaginary      = {N(imag(approx))}")
        print()


# ============================================================
# Example 3: Visualizing convergence in T
# ============================================================

def convergence_data_for_zeta(x=25.7, c=2, T_values=None):
    """
    Return numerical convergence data for F(s) = zeta(s).

    This is useful before plotting. Each entry is a dictionary containing
    T, approximation, exact value, and error.
    """
    if T_values is None:
        T_values = [5, 10, 15, 20, 30, 40, 50, 75, 100]

    exact = exact_sum_for_zeta(x)
    rows = []

    for T in T_values:
        approx = perron_integral(F_zeta, x=x, c=c, T=T)
        rows.append({
            "T": T,
            "approx": real(approx),
            "imaginary": imag(approx),
            "exact": exact,
            "error": real(approx) - exact,
        })

    return rows


def print_convergence_table_for_zeta(x=25.7, c=2, T_values=None):
    """
    Print a simple convergence table as T increases.
    """
    rows = convergence_data_for_zeta(x=x, c=c, T_values=T_values)

    print("=" * 70)
    print("Convergence table for F(s) = zeta(s)")
    print("=" * 70)
    print(f"x = {x}")
    print(f"c = {c}")
    print()
    print("T        approximation        error")
    print("-" * 70)

    for row in rows:
        print(f"{row['T']:<8} {N(row['approx']):<20} {N(row['error'])}")







# ============================================================
# Example 4: F(s) = -zeta'(s)/zeta(s), recovering psi(x)
# ============================================================

def minus_zeta_prime_over_zeta(s, h=1e-6):
    """
    Numerically approximate

        -zeta'(s)/zeta(s).

    This is the Dirichlet series

        sum_{n >= 1} Lambda(n) / n^s

    for Re(s) > 1.

    For now we use a simple centered finite difference for zeta'(s):

        zeta'(s) approx (zeta(s+h) - zeta(s-h))/(2h).

    This is good enough for exploratory Perron experiments on the line
    Re(s) = c > 1.
    """
    zeta_prime_approx = (zeta(s + h) - zeta(s - h)) / (2*h)
    return -zeta_prime_approx / zeta(s)


def exact_sum_for_psi(x):
    """
    Exact target value for F(s) = -zeta'(s)/zeta(s).

    Since

        -zeta'(s)/zeta(s) = sum Lambda(n)/n^s,

    Perron's formula should recover

        psi(x) = sum_{n <= x} Lambda(n).
    """
    return chebyshev_psi(x)


def test_psi_perron(x=100.5, c=2, T_values=[5, 10, 20, 50, 100]):
    """
    Test classical Perron's formula with

        F(s) = -zeta'(s)/zeta(s).

    The target value is psi(x).

    We choose x non-integral, for example x = 100.5, to avoid the jump
    behavior at prime powers.
    """
    exact = exact_sum_for_psi(x)

    print("=" * 70)
    print("Classical Perron test: F(s) = -zeta'(s)/zeta(s)")
    print("=" * 70)
    print(f"x = {x}")
    print(f"c = {c}")
    print(f"Exact psi(x) = {N(exact)}")
    print()

    for T in T_values:
        approx = perron_integral(minus_zeta_prime_over_zeta, x=x, c=c, T=T)

        print(f"T = {T}")
        print(f"  approximation = {N(approx)}")
        print(f"  real part     = {N(real(approx))}")
        print(f"  imaginary     = {N(imag(approx))}")
        print(f"  error         = {N(real(approx) - exact)}")
        print()


def test_psi_perron_at_multiple_x(c=2, T=50):
    """
    Test Perron's formula for psi(x) at several non-integral x-values.
    """
    x_values = [10.5, 25.5, 50.5, 100.5, 200.5]

    print("=" * 70)
    print("Classical Perron test for psi(x) at several x-values")
    print("=" * 70)
    print(f"c = {c}")
    print(f"T = {T}")
    print()

    for x in x_values:
        exact = exact_sum_for_psi(x)
        approx = perron_integral(minus_zeta_prime_over_zeta, x=x, c=c, T=T)

        print(f"x = {x}")
        print(f"  exact psi(x)  = {N(exact)}")
        print(f"  approximation = {N(real(approx))}")
        print(f"  error         = {N(real(approx) - exact)}")
        print(f"  imaginary     = {N(imag(approx))}")
        print()


def convergence_data_for_psi(x=100.5, c=2, T_values=None):
    """
    Return convergence data for Perron's formula applied to psi(x).
    """
    if T_values is None:
        T_values = [5, 10, 15, 20, 30, 40, 50, 75, 100]

    exact = exact_sum_for_psi(x)
    rows = []

    for T in T_values:
        approx = perron_integral(minus_zeta_prime_over_zeta, x=x, c=c, T=T)

        rows.append({
            "T": T,
            "approx": real(approx),
            "imaginary": imag(approx),
            "exact": exact,
            "error": real(approx) - exact,
        })

    return rows


def print_convergence_table_for_psi(x=100.5, c=2, T_values=None):
    """
    Print a convergence table for Perron's formula applied to psi(x).
    """
    rows = convergence_data_for_psi(x=x, c=c, T_values=T_values)

    print("=" * 70)
    print("Convergence table for F(s) = -zeta'(s)/zeta(s)")
    print("=" * 70)
    print(f"x = {x}")
    print(f"c = {c}")
    print(f"Exact psi(x) = {N(exact_sum_for_psi(x))}")
    print()
    print("T        approximation        error")
    print("-" * 70)

    for row in rows:
        print(f"{row['T']:<8} {N(row['approx']):<20} {N(row['error'])}")
    




# ============================================================
# Main script
# ============================================================

test_zeta_perron()
test_zeta_perron_at_multiple_x()
print_convergence_table_for_zeta()

test_psi_perron()
test_psi_perron_at_multiple_x()
print_convergence_table_for_psi()