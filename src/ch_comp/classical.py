
"""
Implementation of classical explicit formula for psi(x).EllipticCurve

For x > 1 not a prime power,

    psi(x) = x - sum_rho x^rho / rho
        - log(2*pi)
        - (1/2) log(1 - x^{-2}),
    
where sum runs over nontrivial zeros rho of the Riemann zeta function.EllipticCurve

For computational experiments and later comparison to Chirre and Helfgott's explicit
formula, we truncate the zero sum at a height T such that RH holds for all zeta zeros
with ordinate below T. That is, for a zeta zero rho with Im rho <= T, Re rho = 1/2.

Thus, for such verified zeros we assume rho = 1/2 + i gamma, and include both conjugate
zeros by taking

    sum_{|gamma| <= T} x^rho / rho = 2 Re sum_{0 < gamma <= T} x^(1/2 + i gamma)/(1/2 + i gamma).

This module is intended for SageMath and requires the odlyzko-zeta zeta zero database.
"""

from sage.all import RealField, ComplexField, pi, log, numerical_approx
import matplotlib.pyplot as plt
import numpy as np

from ch_comp import odlyzko_wrapper as ow, chebyshev

def _validate_x(x):
    """
    Check that x is in the range where the displayed explicit formula makes sense.
    """
    if x <= 1:
        raise ValueError("x must be greater than 1")
    

def single_zero_term(x, gamma, prec=80):
    """
    Helper function to compute a single zero term
    
        x^rho / rho for rho = 1/2 + i gamma

    Here we assume RH holds for rho.

    Parameters
    ----------
    x : real
        Point at which explicit formula is evaluated
    gamma : real
        Positive ordinate of nontrivial zeta zero
    prec : int
        Working precision in bits

    Returns
    -------
    Sage complex number
        x^rho / rho
    """
    _validate_x(x)

    R = RealField(prec)
    C = ComplexField(prec)

    x = R(x)
    rho = C(R(1)/R(2), R(gamma))
    return C(x) ** rho / rho


def explicit_formula_psi_approx(x, gammas, prec=80):
    """
    Compute truncated explicit formula for psi(x) given list of zero ordinates.

    That is, we compute

        x - 2 Re sum_{gamma > 0} x^{1/2 + i gamma}/(1/2 + i gamma)
        - log(2*pi) - (1/2) log(1 - x^{-2})

    where sum runs over gamma in gammas.

    Parameters
    ----------
    x : real
        Point at which to approximate psi(x).
    gammas : iterable, passed as list
        Positive imaginary parts of zeta zeros
    prec : int
        Working precision in bits

    Returns
    -------
    Sage real number
        Truncated explicit-formula approximation to psi(x)
    """

    # First, validate argument x
    _validate_x(x)

    # Construct real field of specified precision and cast x
    R = RealField(prec)
    x = R(x)

    # Compute zero term
    # 2 Re sum_{gamma > 0} x^{1/2 + i gamma}/(1/2 + i gamma)
    C = ComplexField(prec)
    zero_sum = C(0)
    for gamma in gammas:
        zero_sum += single_zero_term(x, gamma, prec=prec)
    zero_term = 2 * zero_sum.real()

    # Compute correction term
    #   log(2*pi) + (1/2) log (1 - x^{-2})
    correction_term = log(2 * R(pi)) + (R(1) / R(2)) * log(1 - x ** (-2))

    # Return final approximation, x - (zero sum) - (correction term)
    return x - zero_term - correction_term




def explicit_formula_psi_approx_first_n(x, n_zeros, prec=80):
    """
    Approximate psi(x) using the first n_zeros positive zeta zero ordinates,
    called from Odlyzko zero database.

    Parameters
    ----------
    x : real
        Point at which to approximate psi(x)
    n_zeros : int
        Number of positive zero ordinates to use
    prec : int
        Working precision in bits

    Returns
    -------
    Sage real number, truncated classical explicit formula approximation to psi(x)
    """
    if n_zeros < 0:
        raise ValueError("n_zeros must be nonnegative")
    gammas = ow.first_zeta_zero_imaginary_parts(n_zeros)
    return explicit_formula_psi_approx(x, gammas, prec=prec)



def explicit_formula_psi_approx_height(x, T, prec=80):
    """
    Approximate psi(x) using all available zeta zero ordinates gamma <= T.

    Parameters
    ----------
    x : real
        Point at which to approximate psi(x)
    T : real
        Height cutoff
    prec : int
        Working precision in bits

    Returns
    -------
    Sage real number, truncated classical explicit formula approximation to psi(x)
    """
    if T < 0:
        raise ValueError("T must be nonnegative")
    gammas = ow.zeta_zeros_up_to_height(T)
    return explicit_formula_psi_approx_height(x, gammas, prec=prec)



"""
print(f"psi(100) = {numerical_approx(chebyshev.chebyshev_psi(100))}")
for n in [1000,2000,5000,10000]:
    print(f"CEF(100,{n}) = {explicit_formula_psi_approx_first_n(100,n)}")
"""

x = np.linspace(2,100,1000)
y1 = [chebyshev.chebyshev_psi(xi) for xi in x]
y2 = [explicit_formula_psi_approx_first_n(xi, 100) for xi in x]

plt.figure(figsize=(8,5))
plt.plot(x,y1,label='$psi(x)$', color='blue',linewidth=2)
plt.plot(x,y2,label='$CEF(100)$',color='red',linewidth=2)

plt.show()
