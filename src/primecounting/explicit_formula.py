"""
Truncated explicit formula experiments for the Chebyshev function psi(x).

The classical explicit formula says, for x > 1 not a prime power,

    psi(x)
    =
    x - sum_rho x^rho / rho
      - log(2*pi)
      - (1/2) log(1 - x^(-2)),

where the sum runs over nontrivial zeros rho of the Riemann zeta function.

For computational experiments, we truncate the zero sum and assume that the
zeros supplied lie on the critical line. Thus, for each positive ordinate
gamma, we use rho = 1/2 + i gamma and include both conjugate zeros by taking

    sum_{|gamma| <= T} x^rho / rho
    =
    2 Re sum_{0 < gamma <= T} x^(1/2 + i gamma)/(1/2 + i gamma).

This module is intended for SageMath.
"""

from sage.all import RealField, ComplexField, pi, log


def _as_real(x, prec=80):
    """
    Convert x to a Sage real number with the given precision.
    """
    R = RealField(prec)
    return R(x)


def _as_complex(z, prec=80):
    """
    Convert z to a Sage complex number with the given precision.
    """
    C = ComplexField(prec)
    return C(z)


def _validate_x(x):
    """
    Check that x is in the range where the displayed explicit formula makes sense.
    """
    if x <= 1:
        raise ValueError("x must be greater than 1")


def zero_from_ordinate(gamma, prec=80):
    """
    Return rho = 1/2 + i gamma.

    This assumes RH for the supplied zero ordinate gamma.

    Parameters
    ----------
    gamma : real
        Positive imaginary part of a nontrivial zeta zero.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage complex number
        rho = 1/2 + i gamma.
    """
    C = ComplexField(prec)
    R = RealField(prec)
    return C(R(1) / R(2), R(gamma))


def single_zero_term(x, gamma, prec=80):
    """
    Compute x^rho / rho for rho = 1/2 + i gamma.

    Parameters
    ----------
    x : real
        Point at which the explicit formula is evaluated.
    gamma : real
        Positive imaginary part of a nontrivial zeta zero.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage complex number
        x^rho / rho.
    """
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

    Parameters
    ----------
    x : real
        Point at which the explicit formula is evaluated.
    gammas : iterable
        Positive imaginary parts of zeta zeros.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage complex number
        The truncated positive-ordinate zero sum.
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

    Parameters
    ----------
    x : real
        Point at which the explicit formula is evaluated.
    gammas : iterable
        Positive imaginary parts of zeta zeros.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage real number
        The truncated zero contribution.
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


def explicit_formula_psi_approx(x, gammas, prec=80):
    """
    Approximate psi(x) using a truncated explicit formula.

    This computes

        x
        - 2 Re sum_{gamma > 0} x^(1/2 + i gamma)/(1/2 + i gamma)
        - log(2*pi)
        - (1/2) log(1 - x^(-2)).

    Parameters
    ----------
    x : real
        Point at which to approximate psi(x).
    gammas : iterable
        Positive imaginary parts of zeta zeros.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage real number
        Truncated explicit-formula approximation to psi(x).
    """
    _validate_x(x)

    R = RealField(prec)
    x = R(x)

    zero_part = zero_contribution_assuming_rh(x, gammas, prec=prec)
    correction = explicit_formula_correction(x, prec=prec)

    return x - zero_part - correction


def _get_first_n_zeta_zeros(n):
    """
    Get the first n zeta zero ordinates from zeta_zeros.py.

    This works in two common situations:
    1. this module is imported as part of the primecounting package;
    2. zeta_zeros.py has already been loaded in a Sage session.
    """
    try:
        from . import zeta_zeros as zz
        return zz.first_zeta_zero_imaginary_parts(n)
    except Exception:
        pass

    if "first_zeta_zero_imaginary_parts" in globals():
        return globals()["first_zeta_zero_imaginary_parts"](n)

    raise RuntimeError(
        "Could not find first_zeta_zero_imaginary_parts. "
        "Either import this as part of the primecounting package, or load "
        "src/primecounting/zeta_zeros.py before using this function."
    )


def _get_zeta_zeros_up_to_height(T):
    """
    Get zeta zero ordinates up to height T from zeta_zeros.py.
    """
    try:
        from . import zeta_zeros as zz
        return zz.zeta_zeros_up_to_height(T)
    except Exception:
        pass

    if "zeta_zeros_up_to_height" in globals():
        return globals()["zeta_zeros_up_to_height"](T)

    raise RuntimeError(
        "Could not find zeta_zeros_up_to_height. "
        "Either import this as part of the primecounting package, or load "
        "src/primecounting/zeta_zeros.py before using this function."
    )


def _get_chebyshev_psi():
    """
    Get chebyshev_psi from chebyshev.py.

    This works either through package import or if chebyshev.py has already
    been loaded in a Sage session.
    """
    try:
        from . import chebyshev as ch
        return ch.chebyshev_psi
    except Exception:
        pass

    if "chebyshev_psi" in globals():
        return globals()["chebyshev_psi"]

    raise RuntimeError(
        "Could not find chebyshev_psi. "
        "Either import this as part of the primecounting package, or load "
        "src/primecounting/chebyshev.py before using this function."
    )


def explicit_formula_psi_approx_first_n(x, n_zeros, prec=80):
    """
    Approximate psi(x) using the first n_zeros positive zeta zero ordinates.

    Parameters
    ----------
    x : real
        Point at which to approximate psi(x).
    n_zeros : int
        Number of positive zero ordinates to use.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage real number
        Truncated explicit-formula approximation to psi(x).
    """
    if n_zeros < 0:
        raise ValueError("n_zeros must be nonnegative")

    gammas = _get_first_n_zeta_zeros(n_zeros)
    return explicit_formula_psi_approx(x, gammas, prec=prec)


def explicit_formula_psi_approx_height(x, T, prec=80):
    """
    Approximate psi(x) using all available zeta zero ordinates gamma <= T.

    Parameters
    ----------
    x : real
        Point at which to approximate psi(x).
    T : real
        Height cutoff.
    prec : int
        Working precision in bits.

    Returns
    -------
    Sage real number
        Truncated explicit-formula approximation to psi(x).
    """
    if T < 0:
        raise ValueError("T must be nonnegative")

    gammas = _get_zeta_zeros_up_to_height(T)
    return explicit_formula_psi_approx(x, gammas, prec=prec)





def compare_with_exact_psi(x, gammas=None, n_zeros=None, T=None, prec=80):
    """
    Compare exact psi(x) with a truncated explicit-formula approximation.

    Exactly one of gammas, n_zeros, or T should be supplied.

    Parameters
    ----------
    x : real
        Point at which to compare.
    gammas : iterable or None
        Positive zeta zero ordinates to use.
    n_zeros : int or None
        Number of first positive zeta zero ordinates to use.
    T : real or None
        Height cutoff for zeta zeros.
    prec : int
        Working precision in bits.

    Returns
    -------
    dict
        Dictionary containing exact value, approximation, and error.
    """
    choices = [gammas is not None, n_zeros is not None, T is not None]
    if sum(choices) != 1:
        raise ValueError("Exactly one of gammas, n_zeros, or T must be supplied.")

    chebyshev_psi = _get_chebyshev_psi()

    if gammas is None:
        if n_zeros is not None:
            gammas = _get_first_n_zeta_zeros(n_zeros)
        else:
            gammas = _get_zeta_zeros_up_to_height(T)

    exact = chebyshev_psi(x)
    approx = explicit_formula_psi_approx(x, gammas, prec=prec)
    error = approx - exact

    return {
        "x": x,
        "n_zeros": len(gammas),
        "exact_psi": exact,
        "approx_psi": approx,
        "error": error,
        "absolute_error": abs(error),
    }


def approximation_error_by_n(x, n_values, prec=80):
    """
    Compute approximation error as the number of zeros varies.

    Parameters
    ----------
    x : real
        Point at which to compare exact psi(x) and the approximation.
    n_values : iterable of ints
        Values of n_zeros to test.
    prec : int
        Working precision in bits.

    Returns
    -------
    list of dict
        One comparison dictionary for each n in n_values.
    """
    results = []

    for n in n_values:
        results.append(compare_with_exact_psi(x, n_zeros=n, prec=prec))

    return results
