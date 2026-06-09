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



# ============================================================
# Chirre-style regularized contour shift
# ============================================================

def zeta_regularized_for_contour(s):
    """
    Regularized zeta function

        F(s) = zeta(s) - 1/(s - 1).

    This removes the pole of zeta at s=1.

    Since numerical integration may sample s=1 exactly, we manually
    return the limiting value Euler's constant there.
    """
    if abs(s - 1) < 1e-10:
        return euler_gamma

    return zeta(s) - 1/(s - 1)


def make_chirre_regularized_integrand(F_regular, x, sigma, T, side):
    """
    Return the Chirre-style regularized integrand

        G(s) = phi_lambda^side((s - 1)/(iT)) F_regular(s) x^s,

    where

        lambda = 2*pi*(sigma - 1)/T.

    On the original vertical line s = 1 + iT*u, this becomes

        phi_lambda^side(u) F_regular(1+iT*u) x^(1+iT*u).

    This is the integrand appearing before the full contour shift.
    """
    if sigma == 1:
        raise ValueError("sigma must not equal 1")

    if T <= 0:
        raise ValueError("T must be positive")

    if side not in {"plus", "minus"}:
        raise ValueError("side must be either 'plus' or 'minus'")

    lam = chirre_lambda(sigma, T)
    phi = make_chirre_phi(lam, side)

    def G(s):
        z = (s - 1) / (I*T)
        return phi(z) * F_regular(s) * x**s

    return G


def chirre_right_vertical_integral(G, T):
    """
    Integral over the original Chirre vertical segment:

        s = 1 + it,  -T <= t <= T.

    Orientation is upward.
    """
    def integrand(t):
        s = 1 + I*t
        return G(s) * I

    real_part = numerical_integral(lambda t: real(integrand(t)), -T, T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), -T, T)[0]

    return real_part + I*imag_part


def chirre_top_horizontal_integral(G, alpha, T):
    """
    Integral over the top horizontal segment:

        s = u + iT,  1 >= u >= alpha.

    Orientation is right-to-left.
    """
    def integrand(u):
        s = u + I*T
        return G(s)

    real_part = numerical_integral(lambda u: real(integrand(u)), 1, alpha)[0]
    imag_part = numerical_integral(lambda u: imag(integrand(u)), 1, alpha)[0]

    return real_part + I*imag_part


def chirre_left_vertical_integral_down(G, alpha, T):
    """
    Integral over the shifted left vertical segment:

        s = alpha + it,  T >= t >= -T.

    Orientation is downward.
    """
    def integrand(t):
        s = alpha + I*t
        return G(s) * I

    real_part = numerical_integral(lambda t: real(integrand(t)), T, -T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), T, -T)[0]

    return real_part + I*imag_part


def chirre_bottom_horizontal_integral(G, alpha, T):
    """
    Integral over the bottom horizontal segment:

        s = u - iT,  alpha <= u <= 1.

    Orientation is left-to-right.
    """
    def integrand(u):
        s = u - I*T
        return G(s)

    real_part = numerical_integral(lambda u: real(integrand(u)), alpha, 1)[0]
    imag_part = numerical_integral(lambda u: imag(integrand(u)), alpha, 1)[0]

    return real_part + I*imag_part


def contour_shift_chirre_regularized_zeta(
    x,
    sigma=0,
    T=20,
    alpha=0.5,
    side="plus",
):
    """
    Numerically test a Chirre-style contour shift using

        F(s) = zeta(s) - 1/(s - 1).

    The integrand is

        G(s) =
            phi_lambda^side((s - 1)/(iT))
            F(s)
            x^s.

    Since F(s) is regular at s=1 and this rectangle crosses no pole,
    the total rectangle integral should be approximately 0.

    This is a mechanics test before introducing residues from zeta zeros.
    """
    if not (alpha < 1):
        raise ValueError("Need alpha < 1.")

    if T <= 0:
        raise ValueError("Need T > 0.")

    G = make_chirre_regularized_integrand(
        F_regular=zeta_regularized_for_contour,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    right = chirre_right_vertical_integral(G, T)
    top = chirre_top_horizontal_integral(G, alpha, T)
    left_down = chirre_left_vertical_integral_down(G, alpha, T)
    bottom = chirre_bottom_horizontal_integral(G, alpha, T)

    rectangle_integral = right + top + left_down + bottom

    right_from_shift = -top - left_down - bottom

    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "alpha": alpha,
        "side": side,
        "right": right,
        "top": top,
        "left_down": left_down,
        "bottom": bottom,
        "rectangle_integral": rectangle_integral,
        "right_from_shift": right_from_shift,
        "right_shift_error": right - right_from_shift,
        "right_normalized": right / (I*T),
        "shift_normalized": right_from_shift / (I*T),
        "top_normalized": top / (I*T),
        "left_normalized": left_down / (I*T),
        "bottom_normalized": bottom / (I*T),
    }


def print_chirre_regularized_contour_shift_result(result):
    """
    Pretty-print the Chirre-style regularized contour-shift diagnostic.
    """
    print("=" * 70)
    print("Chirre-style regularized contour shift: zeta")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"alpha = {result['alpha']}")
    print(f"side = {result['side']}")
    print()

    print("Raw contour pieces:")
    print(f"  right vertical      = {N(result['right'])}")
    print(f"  top horizontal      = {N(result['top'])}")
    print(f"  left vertical down  = {N(result['left_down'])}")
    print(f"  bottom horizontal   = {N(result['bottom'])}")
    print()

    print("No-pole rectangle check:")
    print(f"  rectangle integral  = {N(result['rectangle_integral'])}")
    print()

    print("Shift identity check:")
    print(f"  right vertical      = {N(result['right'])}")
    print(f"  right from shift    = {N(result['right_from_shift'])}")
    print(f"  right shift error   = {N(result['right_shift_error'])}")
    print()

    print("Normalized by iT:")
    print(f"  original vertical   = {N(result['right_normalized'])}")
    print(f"  shifted expression  = {N(result['shift_normalized'])}")
    print(f"  top contribution    = {N(result['top_normalized'])}")
    print(f"  left contribution   = {N(result['left_normalized'])}")
    print(f"  bottom contribution = {N(result['bottom_normalized'])}")
    print()
