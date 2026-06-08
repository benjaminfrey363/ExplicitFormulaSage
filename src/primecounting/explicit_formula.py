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


def _validate_x(x):
    '''
    validate x as an argument for explicit formula
    '''
    if x <= 1:
        raise ValueError("x must be greater than 1")
    

def zero_from_ordinate(gamma, prec=80):
    '''
    Giving an ordinate gamma, return complex number with specified
    precision with real part 1/2 and imaginary part gamma
    '''
    C = ComplexField(prec)
    R = RealField(prec)
    return C(R(1) / R(2), R(gamma))


def single_zero_term(x, gamma, prec=80):
    '''
    Compute x^rho / rho for rho = 1/2 + i gamma
    '''

    _validate_x(x)

    R = RealField(prec)
    C = ComplexField(prec)

    x = R(x)
    rho = zero_from_ordinate(gamma, prec=prec)

    return C(x) ** rho / rho


def positive_zero_sum(x, gammas, prec=80):
    """
    Compute the positive ordinate zero sum

        sum_{gamma > 0} x^(1/2 + i gamma)/(1/2 + i gamma)

    over the supplied list of positive zero ordinates.
    """
    _validate_x(x)

    C = ComplexField(prec)
    total = C(0)

    for gamma in gammas:
        total += single_zero_term(x, gamma, prec=prec)

    return total


def zero_contribution_assuming_rh(x, gammas, prec=80):
    """
    Compute the symmetric zero contribution assuming RH:

        2 Re sum_{gamma > 0} x^(1/2 + i gamma)/(1/2 + i gamma).
    """
    S = positive_zero_sum(x, gammas, prec=prec)
    return 2 * S.real()


def explicit_formula_correction(x, prec=80):
    """
    Compute the nonzero correction term

        log(2*pi) + (1/2) log(1 - x^(-2)).

    In the explicit formula this is subtracted from x together with
    the zero contribution.

    Parameters
    ----------
    x : real
        Point at which the explicit formula is evaluated.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage real number
        The correction term.
    """
    _validate_x(x)

    R = RealField(prec)
    x = R(x)

    return log(2 * R(pi)) + (R(1) / R(2)) * log(1 - x ** (-2))
