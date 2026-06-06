'''
explicit_formula.py

Functions to compute explicit formulae

The classical explicit formula says that for x > 1 not a prime power,

psi(x) = x - sum_rho x^rho / rho
        - log(2*pi)
        - (1/2) log(1 - x^(-2)),

where the sum runs over all nontrivial zeros rho of zeta(s).

For computational experiments, we truncate the zero sum and compute
a sum over zeros for which the Riemann hypothesis has been verified.
Thus, we have rho = 1/2 + i gamma for these zeros, and include both
conjugate zeros:

sum_{|gamma| <= T} x^rho / rho
    = 2 Re sum_{0 < gamma <= T} x^(1/2 + i gamma) / (1/2 + i gamma)
'''

from sage.all import RealField, ComplexField, pi, log


def _as_real(x,prec=80):
    '''
    cast x to a sage real number with given precision
    '''
    R = RealField(prec)
    return R(x)


def _as_complex(z,prec=80):
    '''
    cast z to a sage complex number with given precision
    '''
    C = ComplexField(prec)
    return C(z)