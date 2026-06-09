"""
contour_shift.sage

Toy contour-shift experiments.

This file starts with the classical Perron integrand

    G(s) = zeta(s) x^s / s.

We integrate G(s) around a rectangle with right edge Re(s)=c and
left edge Re(s)=alpha, where

    c > 1,
    0 < alpha < 1.

Inside this rectangle, G(s) has a simple pole at s=1 with residue x.

Therefore, by the residue theorem,

    int_rectangle G(s) ds = 2*pi*i*x.

Equivalently,

    right_vertical
    =
    2*pi*i*x - top - left_down - bottom.

After dividing by 2*pi*i, this gives a numerical contour-shift identity.
"""


# ============================================================
# Numerical complex line integrals
# ============================================================

def complex_integral_over_parametrized_path(path, dpath, a, b):
    """
    Numerically compute

        int_a^b G(path(t)) path'(t) dt

    for a complex-valued path.

    Parameters
    ----------
    path : function
        t |-> complex point s(t).
    dpath : function
        t |-> derivative s'(t).
    a, b : real
        Parameter interval.

    Returns
    -------
    complex
        Numerical line integral.
    """
    def integrand(t):
        s = path(t)
        return perron_zeta_integrand(s) * dpath(t)

    real_part = numerical_integral(lambda t: real(integrand(t)), a, b)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), a, b)[0]

    return real_part + I*imag_part


# ============================================================
# Classical Perron integrand for zeta
# ============================================================

def perron_zeta_integrand(s, x=None):
    """
    Classical Perron integrand

        zeta(s) x^s / s.

    The parameter x is supplied globally by make_perron_zeta_integrand,
    so this default function should usually not be called directly.
    """
    raise RuntimeError(
        "Use make_perron_zeta_integrand(x) to create an integrand."
    )


def make_perron_zeta_integrand(x):
    """
    Return the function

        G(s) = zeta(s) x^s / s.
    """
    return lambda s: zeta(s) * x**s / s


# ============================================================
# Rectangle contour pieces
# ============================================================

def right_vertical_integral(G, c, T):
    """
    Integral over the right vertical edge:

        s = c + it,  -T <= t <= T.

    Orientation is upward.
    """
    def integrand(t):
        s = c + I*t
        return G(s) * I

    real_part = numerical_integral(lambda t: real(integrand(t)), -T, T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), -T, T)[0]

    return real_part + I*imag_part


def top_horizontal_integral(G, alpha, c, T):
    """
    Integral over the top horizontal edge:

        s = u + iT,  c >= u >= alpha.

    Orientation is from right to left.
    """
    def integrand(u):
        s = u + I*T
        return G(s)

    # right to left: integrate u from c down to alpha
    real_part = numerical_integral(lambda u: real(integrand(u)), c, alpha)[0]
    imag_part = numerical_integral(lambda u: imag(integrand(u)), c, alpha)[0]

    return real_part + I*imag_part


def left_vertical_integral_down(G, alpha, T):
    """
    Integral over the left vertical edge:

        s = alpha + it,  T >= t >= -T.

    Orientation is downward.
    """
    def integrand(t):
        s = alpha + I*t
        return G(s) * I

    # downward: integrate t from T down to -T
    real_part = numerical_integral(lambda t: real(integrand(t)), T, -T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), T, -T)[0]

    return real_part + I*imag_part


def bottom_horizontal_integral(G, alpha, c, T):
    """
    Integral over the bottom horizontal edge:

        s = u - iT,  alpha <= u <= c.

    Orientation is from left to right.
    """
    def integrand(u):
        s = u - I*T
        return G(s)

    real_part = numerical_integral(lambda u: real(integrand(u)), alpha, c)[0]
    imag_part = numerical_integral(lambda u: imag(integrand(u)), alpha, c)[0]

    return real_part + I*imag_part


# ============================================================
# Full contour-shift diagnostic
# ============================================================

def contour_shift_zeta_perron(x, c=2, alpha=0.5, T=20):
    """
    Numerically test the rectangular contour shift for

        G(s) = zeta(s) x^s / s.

    The rectangle has vertices

        c - iT,
        c + iT,
        alpha + iT,
        alpha - iT.

    We assume

        c > 1,
        0 < alpha < 1,

    so the only pole crossed is s=1, with residue x.
    """
    if not (c > 1):
        raise ValueError("Need c > 1.")

    if not (0 < alpha < 1):
        raise ValueError("Need 0 < alpha < 1 for this first toy test.")

    if T <= 0:
        raise ValueError("Need T > 0.")

    G = make_perron_zeta_integrand(x)

    right = right_vertical_integral(G, c, T)
    top = top_horizontal_integral(G, alpha, c, T)
    left_down = left_vertical_integral_down(G, alpha, T)
    bottom = bottom_horizontal_integral(G, alpha, c, T)

    rectangle_integral = right + top + left_down + bottom
    residue_contribution = 2*pi*I*x

    right_from_shift = residue_contribution - top - left_down - bottom

    return {
        "x": x,
        "c": c,
        "alpha": alpha,
        "T": T,
        "right": right,
        "top": top,
        "left_down": left_down,
        "bottom": bottom,
        "rectangle_integral": rectangle_integral,
        "residue_contribution": residue_contribution,
        "rectangle_error": rectangle_integral - residue_contribution,
        "right_from_shift": right_from_shift,
        "right_shift_error": right - right_from_shift,
        "perron_right_normalized": right / (2*pi*I),
        "shift_formula_normalized": right_from_shift / (2*pi*I),
        "residue_normalized": x,
        "top_normalized": top / (2*pi*I),
        "left_normalized": left_down / (2*pi*I),
        "bottom_normalized": bottom / (2*pi*I),
    }


def print_contour_shift_result(result):
    """
    Pretty-print the contour-shift diagnostic.
    """
    print("=" * 70)
    print("Toy contour shift: G(s) = zeta(s) x^s / s")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"c = {result['c']}")
    print(f"alpha = {result['alpha']}")
    print(f"T = {result['T']}")
    print()

    print("Raw contour pieces:")
    print(f"  right vertical      = {N(result['right'])}")
    print(f"  top horizontal      = {N(result['top'])}")
    print(f"  left vertical down  = {N(result['left_down'])}")
    print(f"  bottom horizontal   = {N(result['bottom'])}")
    print()

    print("Residue theorem check:")
    print(f"  rectangle integral  = {N(result['rectangle_integral'])}")
    print(f"  2*pi*i*x            = {N(result['residue_contribution'])}")
    print(f"  rectangle error     = {N(result['rectangle_error'])}")
    print()

    print("Shift identity check:")
    print(f"  right vertical      = {N(result['right'])}")
    print(f"  right from shift    = {N(result['right_from_shift'])}")
    print(f"  right shift error   = {N(result['right_shift_error'])}")
    print()

    print("Normalized by 2*pi*i:")
    print(f"  Perron right side   = {N(result['perron_right_normalized'])}")
    print(f"  Shift formula       = {N(result['shift_formula_normalized'])}")
    print(f"  residue term        = {N(result['residue_normalized'])}")
    print(f"  top contribution    = {N(result['top_normalized'])}")
    print(f"  left contribution   = {N(result['left_normalized'])}")
    print(f"  bottom contribution = {N(result['bottom_normalized'])}")
    print()
    