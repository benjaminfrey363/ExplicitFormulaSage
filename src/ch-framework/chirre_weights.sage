"""
chirre_weights.sage

Chirre--Helfgott / Graham--Vaaler weight functions.

This file contains the functions needed to build the majorant and minorant
used in the weighted Perron framework.

The key objects are:

    I_lambda(y)
        the truncated exponential encoding the sharp cutoff;

    phi_lambda^+(t), phi_lambda^-(t)
        compactly supported functions on [-1,1];

    hat{phi_lambda^+}(y), hat{phi_lambda^-}(y)
        their Fourier transforms, computed numerically for now.

The intended inequality is

    hat{phi_lambda^-}(y) <= I_lambda(y) <= hat{phi_lambda^+}(y).

For now, the Fourier transforms are computed numerically using
fourier_transform_compact from weighted_perron.sage.
"""


# ============================================================
# Basic utilities
# ============================================================

def real_sign(x):
    """
    Return the sign of a real number.

    Returns
    -------
    int
        1 if x > 0, -1 if x < 0, and 0 if x = 0.
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def validate_side(side):
    """
    Validate whether side is 'plus' or 'minus'.
    """
    if side not in {"plus", "minus"}:
        raise ValueError("side must be either 'plus' or 'minus'")


def side_epsilon(side):
    """
    Convert side='plus'/'minus' into epsilon=+1/-1.
    """
    validate_side(side)

    if side == "plus":
        return 1

    return -1


# ============================================================
# Truncated exponential I_lambda
# ============================================================

def chirre_lambda(sigma, T):
    """
    Return Chirre--Helfgott's lambda parameter:

        lambda = 2*pi*(sigma - 1)/T.

    Parameters
    ----------
    sigma : real
        Exponent parameter.
    T : real
        Height parameter.

    Returns
    -------
    real
        lambda.
    """
    if T <= 0:
        raise ValueError("T must be positive")

    return 2*pi*(sigma - 1)/T


def truncated_exponential_I_lambda(y, lam):
    """
    Compute Chirre--Helfgott's truncated exponential

        I_lambda(y) = 1_{[0,infty)}(sgn(lambda) y) exp(-lambda y).

    Equivalently:

    - if lambda > 0, then I_lambda(y) = exp(-lambda y) for y >= 0,
      and 0 for y < 0;

    - if lambda < 0, then I_lambda(y) = exp(-lambda y) for y <= 0,
      and 0 for y > 0.

    Parameters
    ----------
    y : real
        Input variable.
    lam : real
        Nonzero lambda parameter.

    Returns
    -------
    real
        I_lambda(y).
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    if real_sign(lam) * y >= 0:
        return exp(-lam*y)

    return 0


def make_I_lambda(lam):
    """
    Return the one-variable function y |-> I_lambda(y).
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    return lambda y: truncated_exponential_I_lambda(y, lam)


# ============================================================
# Chirre--Helfgott / Graham--Vaaler phi functions
# ============================================================

def Phi_circ(nu, z, side):
    """
    Compute Phi^{plus/minus,circ}_nu(z).

    Parameters
    ----------
    nu : positive real
        Positive parameter nu = |lambda|.
    z : real or complex
        Input variable.
    side : str
        Either 'plus' or 'minus'.

    Returns
    -------
    complex
        Phi^{side,circ}_nu(z).
    """
    if nu <= 0:
        raise ValueError("nu must be positive")

    eps = side_epsilon(side)
    w = -2*pi*I*z + nu

    return (1/2) * (coth(w/2) + eps)


def Phi_star(nu, z, side):
    """
    Compute Phi^{plus/minus,star}_nu(z).

    Parameters
    ----------
    nu : positive real
        Positive parameter nu = |lambda|.
    z : real or complex
        Input variable.
    side : str
        Either 'plus' or 'minus'.

    Returns
    -------
    complex
        Phi^{side,star}_nu(z).
    """
    if nu <= 0:
        raise ValueError("nu must be positive")

    eps = side_epsilon(side)
    w = -2*pi*I*z + nu

    return (I/(2*pi)) * (
        (nu/2)*coth(nu/2)
        - (w/2)*coth(w/2)
        + eps*pi*I*z
    )


def chirre_phi_nu(t, nu, side):
    """
    Compute the compactly supported function phi^{plus/minus}_nu(t).

    This function is supported on [-1,1] and is given by

        phi^{plus/minus}_nu(t)
        =
        1_{[-1,1]}(t)
        (
            Phi^{plus/minus,circ}_nu(t)
            + sgn(t) Phi^{plus/minus,star}_nu(t)
        ).

    Parameters
    ----------
    t : real
        Input variable.
    nu : positive real
        Positive parameter.
    side : str
        Either 'plus' or 'minus'.

    Returns
    -------
    complex
        phi^{side}_nu(t).
    """
    if nu <= 0:
        raise ValueError("nu must be positive")

    validate_side(side)

    if abs(t) > 1:
        return 0

    return Phi_circ(nu, t, side) + real_sign(t)*Phi_star(nu, t, side)


def chirre_phi_lambda(t, lam, side):
    """
    Compute Chirre--Helfgott's phi^{plus/minus}_lambda(t).

    It is defined from phi^{plus/minus}_{|lambda|} by

        phi^{plus/minus}_lambda(t)
        =
        phi^{plus/minus}_{|lambda|}(sgn(lambda) t).

    Parameters
    ----------
    t : real
        Input variable.
    lam : real
        Nonzero lambda parameter.
    side : str
        Either 'plus' or 'minus'.

    Returns
    -------
    complex
        phi^{side}_lambda(t).
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    validate_side(side)

    nu = abs(lam)
    return chirre_phi_nu(real_sign(lam)*t, nu, side)


def make_chirre_phi(lam, side):
    """
    Return the one-variable function

        t |-> phi^{plus/minus}_lambda(t).
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    validate_side(side)

    return lambda t: chirre_phi_lambda(t, lam, side)


# ============================================================
# Numerical Fourier transforms of Chirre weights
# ============================================================

def chirre_phi_hat_numeric(y, lam, side):
    """
    Numerically compute hat{phi_lambda^{plus/minus}}(y).

    This requires fourier_transform_compact to already be loaded from
    weighted_perron.sage.

    Parameters
    ----------
    y : real
        Fourier variable.
    lam : real
        Nonzero lambda parameter.
    side : str
        Either 'plus' or 'minus'.

    Returns
    -------
    complex
        Numerical Fourier transform value.
    """
    if "fourier_transform_compact" not in globals():
        raise RuntimeError(
            "fourier_transform_compact is not defined. "
            "Load ch-framework/weighted_perron.sage before chirre_weights.sage."
        )

    phi = make_chirre_phi(lam, side)
    return fourier_transform_compact(phi, y, support=(-1, 1))


def make_chirre_phi_hat_numeric(lam, side):
    """
    Return the one-variable function

        y |-> hat{phi_lambda^{plus/minus}}(y),

    computed by numerical Fourier transform.
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    validate_side(side)

    return lambda y: chirre_phi_hat_numeric(y, lam, side)


# ============================================================
# Diagnostic helpers
# ============================================================

def chirre_weight_values_at_y(y, lam):
    """
    Return minorant, target, and majorant values at y.

    Returns
    -------
    dict
        Dictionary with keys:

        - y
        - minorant_hat
        - I_lambda
        - majorant_hat
        - lower_violation
        - upper_violation
    """
    phi_minus_hat = make_chirre_phi_hat_numeric(lam, "minus")
    phi_plus_hat = make_chirre_phi_hat_numeric(lam, "plus")

    lower = real(phi_minus_hat(y))
    target = truncated_exponential_I_lambda(y, lam)
    upper = real(phi_plus_hat(y))

    return {
        "y": y,
        "minorant_hat": lower,
        "I_lambda": target,
        "majorant_hat": upper,
        "lower_violation": max(0, lower - target),
        "upper_violation": max(0, target - upper),
    }


def chirre_weight_inequality_grid(lam, y_values):
    """
    Test the inequality

        hat{phi^-_lambda}(y) <= I_lambda(y) <= hat{phi^+_lambda}(y)

    on a finite grid of y-values.

    Parameters
    ----------
    lam : real
        Nonzero lambda parameter.
    y_values : list
        Values of y to test.

    Returns
    -------
    dict
        Dictionary containing all rows and worst violations.
    """
    rows = []
    worst_lower_violation = 0
    worst_upper_violation = 0

    for y in y_values:
        row = chirre_weight_values_at_y(y, lam)
        rows.append(row)

        worst_lower_violation = max(
            worst_lower_violation,
            row["lower_violation"],
        )
        worst_upper_violation = max(
            worst_upper_violation,
            row["upper_violation"],
        )

    return {
        "lambda": lam,
        "rows": rows,
        "worst_lower_violation": worst_lower_violation,
        "worst_upper_violation": worst_upper_violation,
    }