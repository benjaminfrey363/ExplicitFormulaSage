"""
weighted_perron.sage

General numerical framework for Chirre--Helfgott's weighted Perron-type
formula.
Builds on top of src/perron

This file implements the identity

    1/(2*pi*i*T) int_{sigma-i∞}^{sigma+i∞}
        phi(Im(s)/T) A(s) x^s ds

    =

    1/(2*pi) sum_n a_n (x/n)^sigma
        phi_hat( T/(2*pi) log(n/x) ).

When phi is supported on [-1,1], the integral is only over

    sigma - iT to sigma + iT.

This corresponds to the general weighted Perron layer, before introducing
Chirre--Helfgott's particular majorants/minorants.
"""


# ============================================================
# Fourier transform helper
# ============================================================

def fourier_transform_compact(phi, y, support=(-1, 1)):
    """
    Numerically compute

        phi_hat(y) = int phi(t) exp(-2*pi*i*y*t) dt

    over the compact support of phi.

    Parameters
    ----------
    phi : function
        Function of one real variable t.
    y : real
        Fourier variable.
    support : tuple
        Integration interval.

    Returns
    -------
    complex
        Numerical Fourier transform value.
    """
    a, b = support

    def integrand(t):
        return phi(t) * exp(-2*pi*I*y*t)

    real_part = numerical_integral(lambda t: real(integrand(t)), a, b)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), a, b)[0]

    return real_part + I*imag_part


# ============================================================
# Weighted Perron identity: integral side
# ============================================================

def weighted_perron_integral(A, phi, x, sigma, T):
    """
    Compute the integral side of the weighted Perron formula:

        1/(2*pi*i*T) int_{sigma-iT}^{sigma+iT}
            phi(Im(s)/T) A(s) x^s ds.

    Parametrize

        s = sigma + i*T*u,  -1 <= u <= 1.

    Then ds = i*T du, so the integral becomes

        1/(2*pi) int_{-1}^{1}
            phi(u) A(sigma+i*T*u) x^(sigma+i*T*u) du.
    """
    def integrand(u):
        s = sigma + I*T*u
        return phi(u) * A(s) * x**s

    real_part = numerical_integral(lambda u: real(integrand(u)), -1, 1)[0]
    imag_part = numerical_integral(lambda u: imag(integrand(u)), -1, 1)[0]

    return (real_part + I*imag_part) / (2*pi)


# ============================================================
# Weighted Perron identity: sum side
# ============================================================

def weighted_perron_sum(a, phi_hat, x, sigma, T, N_max):
    """
    Compute a truncated version of the sum side:

        1/(2*pi) sum_n a_n (x/n)^sigma
            phi_hat( T/(2*pi) log(n/x) ).

    Parameters
    ----------
    a : function
        Coefficient function n |-> a_n.
    phi_hat : function
        Fourier transform of phi.
    x : real
        Scale parameter.
    sigma : real
        Vertical line real part.
    T : real
        Height parameter.
    N_max : int
        Truncation point for the n-sum.

    Returns
    -------
    complex
        Truncated weighted sum.
    """
    total = 0

    for n in range(1, N_max + 1):
        y = T/(2*pi) * log(n/x)
        total += a(n) * (x/n)**sigma * phi_hat(y)

    return total / (2*pi)


# ============================================================
# Comparison helper
# ============================================================

def compare_weighted_perron(A, a, phi, phi_hat, x, sigma, T, N_max):
    """
    Compare the integral side and the truncated sum side.

    This is mainly a diagnostic helper. Agreement improves as N_max
    increases, assuming the n-sum has been truncated far enough.
    """
    integral_side = weighted_perron_integral(A, phi, x, sigma, T)
    sum_side = weighted_perron_sum(a, phi_hat, x, sigma, T, N_max)

    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "N_max": N_max,
        "integral_side": integral_side,
        "sum_side": sum_side,
        "difference": integral_side - sum_side,
        "absolute_difference": abs(integral_side - sum_side),
    }