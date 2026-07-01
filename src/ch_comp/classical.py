
"""
Implementation of classical explicit formula for psi(x).

For x > 1 not a prime power,

    psi(x) = x - sum_rho x^rho / rho
        - log(2*pi)
        - (1/2) log(1 - x^{-2}),
    
where sum runs over nontrivial zeros rho of the Riemann zeta function.

For computational experiments and later comparison to Chirre and Helfgott's explicit
formula, we truncate the zero sum at a height T such that RH holds for all zeta zeros
with ordinate below T. That is, for a zeta zero rho with Im rho <= T, Re rho = 1/2.

Thus, for such verified zeros we assume rho = 1/2 + i gamma, and include both conjugate
zeros by taking

    sum_{|gamma| <= T} x^rho / rho = 2 Re sum_{0 < gamma <= T} x^(1/2 + i gamma)/(1/2 + i gamma).

This module is intended for SageMath and requires the odlyzko-zeta zeta zero database.
"""

from sage.all import RealField, ComplexField, pi, log, numerical_approx, exp
import matplotlib.pyplot as plt
import numpy as np
import time

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
    return explicit_formula_psi_approx(x, gammas, prec=prec)







"""
Class-based refactor
"""


class ClassicalExplicitFormula:
    """
    Evaluator for the truncated classical explicit formula for psi(x).

    Refactored to rebuilding fields, rho-values, and denominators for every x.
    """

    def __init__(self, gammas, prec=80):
        self.prec = prec
        self.R = RealField(prec)
        self.C = ComplexField(prec)

        # constants
        self.half = self.R(1) / self.R(2)
        self.log_2pi = log(2 * self.R(pi))

        self.gammas = [self.R(gamma) for gamma in gammas]
        self.rhos = [self.C(self.half, gamma) for gamma in self.gammas]
        self.inv_rhos = [1 / rho for rho in self.rhos]

    def psi(self, x):
        """
        Evaluate the truncated classical explicit formula at one x.
        """
        _validate_x(x)

        x = self.R(x)
        log_x = log(x)

        zero_sum = self.C(0)
        for rho, inv_rho in zip(self.rhos, self.inv_rhos):
            zero_sum += exp(rho * log_x) * inv_rho

        zero_term = 2 * zero_sum.real()
        correction_term = self.log_2pi + self.half * log(1 - x ** (-2))

        return x - zero_term - correction_term



def compute_cef(n_zeros, lb, ub, n_points, scale="linear", prec=80):
    """
    Evaluate truncated classical explicit formula for psi(x) on range of points
    Optimized to reduce redundant construction of fields and loading of zeros

    Parameters
    ----------
    n_zeros : int
        Number of zeta zeros used in approximation
    lb : int (for now)
        Lower bound of range of approximation
    ub : int (for now)
        Upper bound of range of approximation
    n_points : int
        Number of points sampled in range
    scale : "linear" || "log"
        Scale at which to sample points

    Returns
    -------
    exec_time : float
        Execution time
    xs : list of floats
        Range of input values
    ys : list of sage real numbers
        CEF(n_zeros, x) for x in range
    """
    # Start timer
    start_time = time.perf_counter()

    # construct range
    if scale not in ["linear", "log"]:
        raise ValueError("scale must be linear or log.")
    if scale == "linear":
        xs = np.linspace(lb, ub, n_points)
    else:
        xs = np.logspace(np.log10(lb), np.log10(ub), n_points)
    
    # load zeros
    gammas = ow.first_zeta_zero_imaginary_parts(n_zeros)
    # construct CEF and compute
    cef = ClassicalExplicitFormula(gammas, prec=prec)
    ys = [cef.psi(x) for x in xs]

    # stop timer
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    
    return exec_time, xs, ys


# Approximate L2 distance between 2 functions on pre-specified range xs
# Take as arg values of fns on this range, y1s and y2s
def _l2dist_approx(y1s, y2s):
    if len(y1s) != len(y2s):
        raise ValueError("Function output arrays must have the same size")
    return sum([ abs(y1 - y2)**2 for (y1,y2) in zip(y1s,y2s)])



def cef_error(n_zeros, lb, ub, n_points, prec=80):
    """
    Compute L2 error of classical explicit formula vs actual psi(x) on specified range

    Parameters
    ----------
    n_zeros : int
        Number of zeta zeros used in approximation
    lb : int (for now)
        Lower bound of range of approximation
    ub : int (for now)
        Upper bound of range of approximation
    n_points : int
        Number of points sampled in range

    Returns
    -------
    error : float
        Computed error || CEF(x) - psi(x) ||
    """
    xs = np.linspace(lb,ub,n_points)
    psis = [chebyshev.chebyshev_psi(x) for x in xs]
    gammas = ow.first_zeta_zero_imaginary_parts(n_zeros)
    cef = ClassicalExplicitFormula(gammas,prec=prec)
    ys = [cef.psi(x) for x in xs]
    return numerical_approx(_l2dist_approx(psis, ys),prec)


"""
print(cef_error(10, 2, 100, 1000))
print(cef_error(20, 2, 100, 1000))
print(cef_error(100, 2, 100, 1000))
print(cef_error(1000, 2, 100, 1000))
"""


"""
Demo: compute plot

start_time = time.perf_counter()

x = np.linspace(2,100,1000)
y1 = [chebyshev.chebyshev_psi(xi) for xi in x]

gammas = ow.first_zeta_zero_imaginary_parts(1000)
# compute CEF(100), CEF(500), CEF(1000)
cef100 = ClassicalExplicitFormula(gammas[0:100], prec=80)
cef500 = ClassicalExplicitFormula(gammas[0:500], prec=80)
cef1000 = ClassicalExplicitFormula(gammas, prec=80)
y100 = [cef100.psi(xi) for xi in x]
y500 = [cef500.psi(xi) for xi in x]
y1000 = [cef1000.psi(xi) for xi in x]

end_time = time.perf_counter()

plt.figure(figsize=(8,5))
plt.plot(x,y1,label="$psi(x)$",color="black",linewidth=2)
plt.plot(x,y100,label='CEF(100)', color='blue',linewidth=2)
plt.plot(x,y500,label='CEF(500)',color='purple',linewidth=2)
plt.plot(x,y1000,label='CEF(1000)', color='red',linewidth=2)

plt.show()

exec_time = end_time - start_time
print(f"Elapsed time: {exec_time}")
"""



