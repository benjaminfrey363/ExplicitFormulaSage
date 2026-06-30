
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

from sage.all import ComplexField

"""
Weight computations (omega)
"""

def theta(T):
    return lambda s: 1 - s / ( I*T)
