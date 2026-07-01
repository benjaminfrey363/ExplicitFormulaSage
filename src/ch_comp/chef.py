
"""
Implementation of Chirre and Helfgott's EF for psi(x)/x

    psi(x)/x = (pi/T) coth (pi/T) - log(2pi)/x + (pi/T) sum_k (2 pi k/ T) x^{-2k-1}
    - 2pi/T Im ( sum_rho m_rho omega_T^+(rho) x^{rho - 1} )
    + Error term

Sum over k is over all positive integers k
Sum over rho is over rho in Z_A^+(T), that is, non-trivial zeros of the
zeta function up to imaginary height T

For purposes here, we approximate psi(x) via main terms for different
values of T, and truncate the sum over k as well.
Sum over k tends to  -(1/2) log(1 - x^{-2}) for T large, and first term
(pi/T) coth(pi/T) tends to 1 for T large
"""

from sage.all import RealField, ComplexField, cot, pi, coth, numerical_approx, log
from ch_comp import odlyzko_wrapper as ow, chebyshev
import time, numpy as np
import matplotlib.pyplot as plt

def _validate_x(x):
    """
    Check that x is in the range where the displayed explicit formula makes sense.
    """
    if x <= 1:
        raise ValueError("x must be greater than 1")

"""
Weight computations (omega)

For the specialization to Chebyshev, we will always have sigma = 0,
except for a theta_{T,1} term in error (not used atp)
"""

def theta_Tsigma(T, sigma, prec=80):
    R = RealField(prec)
    C = ComplexField(prec)

    T = R(T)
    sigma = R(sigma)
    I = C.gen()

    if T <= 0:
        raise ValueError("T must be positive")

    def theta(s):
        s = C(s)
        return C(1) - (s - C(sigma))/(I*C(T))
    return theta

def c_Tsigma(T, sigma, prec=80):
    R = RealField(prec)
    C = ComplexField(prec)

    theta = theta_Tsigma(T,sigma, prec=prec)
    T = R(T); I = C.gen()
    theta_eval = theta(C(1) + I*C(T))
    return theta_eval * cot(C(pi) * theta_eval)

def omega_plus_Tsigma(T, sigma, prec=80):
    C = ComplexField(prec)

    theta = theta_Tsigma(T,sigma,prec=prec)
    c = c_Tsigma(T,sigma,prec=prec)

    def omega(s):
        z = theta(s)
        return -z * cot(C(pi) * z) + c
    
    return omega


"""
First term (pi/T) coth (pi/T)
"""

def coth_term_T(T, prec=80):
    R = RealField(prec)
    T = R(T)
    return (R(pi) / T) * coth(R(pi) / T)


"""
Second term (pi/T) sum_k (2 pi k/T) x^{-2k-1}
Specify truncation height of infinite sum, could later replace with limit
Return log term, a function of x
"""

def logterm_TN(T, N, prec=80):
    R = RealField(prec)
    T = R(T)

    def logterm(x):
        x = R(x)
        return (R(pi) / T) * sum(
            (2 * R(pi) * R(k) / T) * x ** (-2 * k - 1)
            for k in range(1, N + 1)
        )
    return logterm


class CHExplicitFormula:
    """
    Evaluator for truncated CH explicit formula for psi(x)

    Implemented in class to avoid rebuilding fields and loading zeros
    """

    def __init__(self, gammas, T, N, prec=80):
        self.prec = prec
        self.R = RealField(prec)
        self.C = ComplexField(prec)

        self.half = self.R(1) / self.R(2)

        self.T = self.R(T)
        self.N = N
        if self.T <= 0:
            raise ValueError("T must be positive")
        self.gammas = [self.R(gamma) for gamma in gammas]
        if self.gammas and max(self.gammas) > self.T:
            raise ValueError(
                f"Received a zero with gamma={max(self.gammas)} > T={self.T}. "
                "For the height-T formula, pass only zeros with gamma <= T."
            )

        # Pre-compute weights and main terms
        self.omega = omega_plus_Tsigma(T,0)
        self.coth_term = coth_term_T(T)
        self.log_term = logterm_TN(T,N)     # function of x
        self.constant_correction = -log(2 * self.R(pi))

        self.rhos = [self.C(self.half, gamma) for gamma in self.gammas]

    def psi(self, x):
        """
        Evaluate CEF, truncated at height T, at one x.

        psi(x)/x = (pi/T) coth (pi/T) - log(2pi)/x + ((pi/T) sum_k (2 pi k/ T) x^{-2k-1})
        - 2pi/T Im ( sum_rho m_rho omega_T^+(rho) x^{rho - 1} )

        Compute approximation of psi(x)/x and multiply by x
        """
        _validate_x(x)

        x = self.R(x)

        zero_sum = self.C(0)
        for rho in self.rhos:
            zero_sum += self.omega(rho) * x**(rho - 1)
        zero_term = (-2)*(pi/self.T)*zero_sum.imag()
        main_term = self.coth_term + self.log_term(x)

        form = main_term + zero_term
        return x*form + self.constant_correction
    



# Approximate L2 distance between 2 functions on pre-specified range xs
# Take as arg values of fns on this range, y1s and y2s
def _l2dist_approx(y1s, y2s):
    if len(y1s) != len(y2s):
        raise ValueError("Function output arrays must have the same size")
    return sum([ abs(y1 - y2)**2 for (y1,y2) in zip(y1s,y2s)])




def compute_chef_T(T, lb, ub, n_points, scale="linear", prec=80):
    """
    Evaluate truncated CH explicit formula for psi(x) on range of points
    Optimized to reduce redundant construction of fields and loading of zeros

    Parameters
    ----------
    T : float >= 0
        Truncation height, load zeros below this positive height
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
        CHEF(T, x) for x in range
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
    gammas = ow.zeta_zeros_up_to_height(T)
    # for now hardcode
    N = 1000
    # construct CEF and compute
    cef = CHExplicitFormula(gammas, T, N, prec=prec)
    ys = [cef.psi(x) for x in xs]

    # stop timer
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    
    return exec_time, xs, ys




# Wrapper to make consistent with classical EF, pass n_zeros rather than
# truncation height T
def compute_chef(n_zeros, lb, ub, n_points, prec=80):
    """
    Evaluate truncated CH explicit formula for psi(x) on range of points
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

    Returns
    -------
    exec_time : float
        Execution time
    xs : list of floats
        Range of input values
    ys : list of sage real numbers
        CHEF(T, x) for x in range
    """
    # Start timer
    start_time = time.perf_counter()

    # construct range
    xs = np.linspace(lb, ub, n_points)
    
    # load zeros
    gammas = ow.first_zeta_zero_imaginary_parts(n_zeros)
    T = max(gammas)
    # for now hardcode
    N = 1000
    # construct CEF and compute
    cef = CHExplicitFormula(gammas, T, N, prec=prec)
    ys = [cef.psi(x) for x in xs]

    # stop timer
    end_time = time.perf_counter()
    exec_time = end_time - start_time
    
    return exec_time, xs, ys





"""
exec_time, xs, ys = compute_chef(1000,2,100,1000)

psis = [chebyshev.chebyshev_psi(x) for x in xs]

plt.figure(figsize=(8,5))
plt.plot(xs,psis,label="psi(x)",color="black",linewidth=2)
plt.plot(xs,ys,label="CHEF(1000)",color="red",linewidth=2)

plt.show()
"""



