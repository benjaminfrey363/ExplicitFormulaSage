
"""
Implementation of Chirre and Helfgott's EF for psi(x)/x

    psi(x)/x = (pi/T) coth (pi/T) + (pi/T) sum_k (2 pi k/ T) x^{-2k-1}
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

from sage.all import RealField, ComplexField, I, cot, pi, coth, numerical_approx
from ch_comp import odlyzko_wrapper as ow, chebyshev


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

def theta_Tsigma(T, sigma):
    def theta(s):
        return 1 - (s - sigma)/(I*T)
    return theta

def c_Tsigma(T, sigma):
    theta = theta_Tsigma(T,sigma)
    theta_eval = theta(1 + I*T)
    return theta_eval * cot(pi * theta_eval)

def omega_plus_Tsigma(T, sigma):
    theta = theta_Tsigma(T,sigma)
    c = c_Tsigma(T,sigma)
    def omega(s):
        return (-1)*theta(s)*cot(pi*theta(s)) + c
    return omega


"""
First term (pi/T) coth (pi/T)
"""

def coth_term_T(T):
    return (pi/T) * coth(pi/T)


"""
Second term (pi/T) sum_k (2 pi k/T) x^{-2k-1}
Specify truncation height of infinite sum, could later replace with limit
Return log term, a function of x
"""

def logterm_TN(T, N):
    def logterm(x):
        truncated_sum = sum([(2*pi*k/T)*(x**(-2*k - 1)) for k in range(1,N+1)])
        return (pi/T)*truncated_sum
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

        self.gammas = [self.R(gamma) for gamma in gammas]
        self.T = T
        # Pre-compute weights and main terms
        self.omega = omega_plus_Tsigma(T,0)
        self.coth_term = coth_term_T(T)
        self.log_term = logterm_TN(T,N)     # function of x
        self.rhos = [self.C(self.half, gamma) for gamma in self.gammas]

    def psi(self, x):
        """
        Evaluate CEF, truncated at height T, at one x.

        psi(x)/x = (pi/T) coth (pi/T) + ((pi/T) sum_k (2 pi k/ T) x^{-2k-1})
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
        return x*form
    

gammas = ow.zeta_zeros_up_to_height(100)
cef = CHExplicitFormula(gammas, 100, 100)
print(numerical_approx(cef.psi(100),80))
print(numerical_approx(chebyshev.chebyshev_psi(100),80))

gammas = ow.zeta_zeros_up_to_height(1000)
cef = CHExplicitFormula(gammas, 1000, 100)
print(numerical_approx(cef.psi(100),80))
print(numerical_approx(chebyshev.chebyshev_psi(100),80))







