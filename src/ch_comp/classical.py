
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

from sage.all import pi, log
from ch_comp import odlyzko_wrapper as ow

print(ow.first_zeta_zero_imaginary_parts(10))




