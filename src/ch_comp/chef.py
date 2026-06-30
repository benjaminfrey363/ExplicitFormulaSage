
"""
Implementation of Chirre and Helfgott's EF for psi(x)/x

    psi(x)/x = (pi/T) coth (pi/T) + (pi/T) sum_k (2 pi k/ T) x^{-2k-1}
    - 2pi/T Im ( sum_rho ) m_rho omega_T^+(rho) x^{rho - 1}
    + Error term

Sum over k is over all positive integers k
Sum over rho is over rho in Z_A^+(T), that is, non-trivial zeros of the
zeta function up to imaginary height T

For purposes here, we approximate psi(x) via main terms for different
values of T, and truncate the sum over k as well.
Sum over k tends to  -(1/2) log(1 - x^{-2}), also provided option to
use this limit rather than the truncation
"""

from sage.all import ComplexField, I, cot, pi

"""
Weight computations (omega)
"""

def theta_Tsigma(T, sigma):
    def theta(s):
        return 1 - (s - sigma)/(I*T)
    return theta

def c_Tsigma(T, sigma):
    theta = theta_Tsigma(T,sigma)
    theta_eval = theta(1 + I*T)
    return theta_eval * cot(pi * theta_eval)

def ometa_plus_Tsigma(T, sigma):
    theta = theta_Tsigma(T,sigma)
    c = c_Tsigma(T,sigma)
    def omega(s):
        return (-1)*theta(s)*cot(pi*theta(s)) + c
    return omega


"""
Approximation of main term sum by truncating
    Add later alternative to replace with limit -(1/2) log(1 - x^{-2})
"""


