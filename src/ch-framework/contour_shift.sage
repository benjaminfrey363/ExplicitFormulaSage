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



def make_chirre_regularized_integrand_pieces(F_regular, x, sigma, T, side):
    """
    Return G_circ and G_star such that on the original vertical segment

        G(s) = G_circ(s) + sgn(Im(s)) G_star(s).

    This mirrors Chirre--Helfgott's contour-shift setup.
    """
    lam = chirre_lambda(sigma, T)

    def G_circ(s):
        z = (s - 1) / (I*T)
        return chirre_Phi_circ_lambda(z, lam, side) * F_regular(s) * x**s

    def G_star(s):
        z = (s - 1) / (I*T)
        return chirre_Phi_star_lambda(z, lam, side) * F_regular(s) * x**s

    return G_circ, G_star




# ============================================================
# Chirre-style contour shift using circ/star pieces
# ============================================================

def chirre_original_vertical_integral_from_pieces(G_circ, G_star, T):
    """
    Compute the original Chirre vertical integral using the decomposition

        G(s) = G_circ(s) + sgn(Im(s)) G_star(s)

    on the segment s = 1 + it, -T <= t <= T.

    Thus, for t > 0 we use G_circ + G_star, and for t < 0 we use
    G_circ - G_star.

    We split the integral at t = 0 because of the sign change.
    """
    def integrand_lower(t):
        s = 1 + I*t
        return (G_circ(s) - G_star(s)) * I

    def integrand_upper(t):
        s = 1 + I*t
        return (G_circ(s) + G_star(s)) * I

    real_lower = numerical_integral(lambda t: real(integrand_lower(t)), -T, 0)[0]
    imag_lower = numerical_integral(lambda t: imag(integrand_lower(t)), -T, 0)[0]

    real_upper = numerical_integral(lambda t: real(integrand_upper(t)), 0, T)[0]
    imag_upper = numerical_integral(lambda t: imag(integrand_upper(t)), 0, T)[0]

    return (real_lower + real_upper) + I*(imag_lower + imag_upper)


def chirre_circ_rectangle_integrals(G_circ, alpha, T):
    """
    Compute the rectangle contour pieces for G_circ.

    G_circ is meromorphic/holomorphic in the rectangle, so this behaves
    like an ordinary contour-shift term.
    """
    right = chirre_right_vertical_integral(G_circ, T)
    top = chirre_top_horizontal_integral(G_circ, alpha, T)
    left_down = chirre_left_vertical_integral_down(G_circ, alpha, T)
    bottom = chirre_bottom_horizontal_integral(G_circ, alpha, T)

    return {
        "right": right,
        "top": top,
        "left_down": left_down,
        "bottom": bottom,
        "rectangle": right + top + left_down + bottom,
    }


def chirre_star_upper_rectangle_integrals(G_star, alpha, T):
    """
    Compute the upper-half rectangle pieces for G_star.

    This rectangle has vertices:

        1,
        1 + iT,
        alpha + iT,
        alpha.

    Orientation:
        right vertical upward from 1 to 1+iT,
        top right-to-left,
        left vertical downward from alpha+iT to alpha,
        seam along real axis from alpha to 1.

    The seam term is important: it is part of the contour boundary.
    """
    # right vertical: s = 1 + it, 0 <= t <= T
    def right_integrand(t):
        s = 1 + I*t
        return G_star(s) * I

    right_real = numerical_integral(lambda t: real(right_integrand(t)), 0, T)[0]
    right_imag = numerical_integral(lambda t: imag(right_integrand(t)), 0, T)[0]
    right = right_real + I*right_imag

    # top: s = u + iT, 1 >= u >= alpha
    top = chirre_top_horizontal_integral(G_star, alpha, T)

    # left vertical down: s = alpha + it, T >= t >= 0
    def left_integrand(t):
        s = alpha + I*t
        return G_star(s) * I

    left_real = numerical_integral(lambda t: real(left_integrand(t)), T, 0)[0]
    left_imag = numerical_integral(lambda t: imag(left_integrand(t)), T, 0)[0]
    left_down = left_real + I*left_imag

    # seam: s = u, alpha <= u <= 1
    def seam_integrand(u):
        s = u
        return G_star(s)

    seam_real = numerical_integral(lambda u: real(seam_integrand(u)), alpha, 1)[0]
    seam_imag = numerical_integral(lambda u: imag(seam_integrand(u)), alpha, 1)[0]
    seam = seam_real + I*seam_imag

    return {
        "right": right,
        "top": top,
        "left_down": left_down,
        "seam": seam,
        "rectangle": right + top + left_down + seam,
    }


def chirre_star_lower_rectangle_integrals(G_star, alpha, T):
    """
    Compute the lower-half rectangle pieces for G_star.

    This rectangle has vertices:

        1,
        alpha,
        alpha - iT,
        1 - iT.

    Orientation is chosen so that the right vertical piece is upward
    from 1-iT to 1, matching the original vertical orientation.

    Boundary pieces:
        right vertical upward from 1-iT to 1,
        seam along real axis from 1 to alpha,
        left vertical downward from alpha to alpha-iT,
        bottom from alpha-iT to 1-iT.
    """
    # right vertical: s = 1 + it, -T <= t <= 0
    def right_integrand(t):
        s = 1 + I*t
        return G_star(s) * I

    right_real = numerical_integral(lambda t: real(right_integrand(t)), -T, 0)[0]
    right_imag = numerical_integral(lambda t: imag(right_integrand(t)), -T, 0)[0]
    right = right_real + I*right_imag

    # seam: s = u, 1 >= u >= alpha
    def seam_integrand(u):
        s = u
        return G_star(s)

    seam_real = numerical_integral(lambda u: real(seam_integrand(u)), 1, alpha)[0]
    seam_imag = numerical_integral(lambda u: imag(seam_integrand(u)), 1, alpha)[0]
    seam = seam_real + I*seam_imag

    # left vertical down: s = alpha + it, 0 >= t >= -T
    def left_integrand(t):
        s = alpha + I*t
        return G_star(s) * I

    left_real = numerical_integral(lambda t: real(left_integrand(t)), 0, -T)[0]
    left_imag = numerical_integral(lambda t: imag(left_integrand(t)), 0, -T)[0]
    left_down = left_real + I*left_imag

    # bottom: s = u - iT, alpha <= u <= 1
    bottom = chirre_bottom_horizontal_integral(G_star, alpha, T)

    return {
        "right": right,
        "seam": seam,
        "left_down": left_down,
        "bottom": bottom,
        "rectangle": right + seam + left_down + bottom,
    }


def contour_shift_chirre_pieces_regularized_zeta(
    x,
    sigma=0,
    T=20,
    alpha=0.5,
    side="plus",
):
    """
    Numerically test the Chirre-style contour shift using the decomposition

        G = G_circ + sgn(Im s) G_star.

    We use

        F(s) = zeta(s) - 1/(s - 1),

    so there should be no pole contribution in the rectangle.

    This test checks three things:

    1. G_circ rectangle integral is approximately 0.
    2. G_star upper and lower rectangle integrals are approximately 0.
    3. The original vertical integral reconstructed from the pieces agrees
       with the shifted expression including seam terms.
    """
    if not (alpha < 1):
        raise ValueError("Need alpha < 1.")

    if T <= 0:
        raise ValueError("Need T > 0.")

    G_circ, G_star = make_chirre_regularized_integrand_pieces(
        F_regular=zeta_regularized_for_contour,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    original = chirre_original_vertical_integral_from_pieces(G_circ, G_star, T)

    circ = chirre_circ_rectangle_integrals(G_circ, alpha, T)
    star_upper = chirre_star_upper_rectangle_integrals(G_star, alpha, T)
    star_lower = chirre_star_lower_rectangle_integrals(G_star, alpha, T)

    # For the circ piece:
    #   right + top + left_down + bottom = 0
    # so
    #   right = -top - left_down - bottom.
    circ_shifted_right = -circ["top"] - circ["left_down"] - circ["bottom"]

    # For the star upper piece:
    #   right + top + left_down + seam = 0
    # so
    #   right = -top - left_down - seam.
    star_upper_shifted_right = (
        -star_upper["top"]
        -star_upper["left_down"]
        -star_upper["seam"]
    )

    # For the star lower piece:
    #   right + seam + left_down + bottom = 0
    # so
    #   right = -seam - left_down - bottom.
    star_lower_shifted_right = (
        -star_lower["seam"]
        -star_lower["left_down"]
        -star_lower["bottom"]
    )

    # Original vertical is:
    #
    #   circ right over full segment
    #   + star upper right
    #   - star lower right
    #
    # because on the lower half the original integrand is G_circ - G_star.
    shifted = (
        circ_shifted_right
        + star_upper_shifted_right
        - star_lower_shifted_right
    )

    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "alpha": alpha,
        "side": side,
        "original": original,
        "shifted": shifted,
        "shift_error": original - shifted,
        "circ_rectangle": circ["rectangle"],
        "star_upper_rectangle": star_upper["rectangle"],
        "star_lower_rectangle": star_lower["rectangle"],
        "circ": circ,
        "star_upper": star_upper,
        "star_lower": star_lower,
        "original_normalized": original / (I*T),
        "shifted_normalized": shifted / (I*T),
        "shift_error_normalized": (original - shifted) / (I*T),
    }


def print_chirre_pieces_contour_shift_result(result):
    """
    Pretty-print the circ/star Chirre contour-shift diagnostic.
    """
    print("=" * 70)
    print("Chirre circ/star contour shift: regularized zeta")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"alpha = {result['alpha']}")
    print(f"side = {result['side']}")
    print()

    print("Rectangle checks:")
    print(f"  circ rectangle       = {N(result['circ_rectangle'])}")
    print(f"  star upper rectangle = {N(result['star_upper_rectangle'])}")
    print(f"  star lower rectangle = {N(result['star_lower_rectangle'])}")
    print()

    print("Shift identity:")
    print(f"  original vertical    = {N(result['original'])}")
    print(f"  shifted expression   = {N(result['shifted'])}")
    print(f"  shift error          = {N(result['shift_error'])}")
    print()

    print("Normalized by iT:")
    print(f"  original vertical    = {N(result['original_normalized'])}")
    print(f"  shifted expression   = {N(result['shifted_normalized'])}")
    print(f"  shift error          = {N(result['shift_error_normalized'])}")
    print()





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





# ============================================================
# Chirre circ/star contour shift with a toy pole
# ============================================================

def make_toy_pole_function(rho):
    """
    Return the meromorphic toy function

        F(s) = 1/(s - rho).

    This has a simple pole at s = rho with residue 1.
    """
    return lambda s: 1/(s - rho)


def is_inside_chirre_rectangle(rho, alpha, T):
    """
    Check whether rho lies inside the rectangle

        alpha <= Re(s) <= 1,
        -T <= Im(s) <= T.
    """
    return (
        alpha < real(rho) < 1
        and -T < imag(rho) < T
    )


def chirre_toy_residue_circ(rho, x, sigma, T, side):
    """
    Residue of

        G_circ(s) =
            Phi_circ_lambda((s-1)/(iT)) * x^s / (s-rho)

    at s = rho.
    """
    lam = chirre_lambda(sigma, T)
    z = (rho - 1) / (I*T)

    return chirre_Phi_circ_lambda(z, lam, side) * x**rho


def chirre_toy_residue_star(rho, x, sigma, T, side):
    """
    Residue of

        G_star(s) =
            Phi_star_lambda((s-1)/(iT)) * x^s / (s-rho)

    at s = rho.
    """
    lam = chirre_lambda(sigma, T)
    z = (rho - 1) / (I*T)

    return chirre_Phi_star_lambda(z, lam, side) * x**rho


def chirre_toy_residue_for_original_integrand(rho, x, sigma, T, side):
    """
    Residue contribution for the original piecewise integrand

        G_circ + sgn(Im(s)) G_star.

    If Im(rho) > 0, use circ + star.
    If Im(rho) < 0, use circ - star.
    If Im(rho) = 0, this toy function is on the seam, and we do not
    handle it in this diagnostic.
    """
    if abs(imag(rho)) < 1e-12:
        raise ValueError("rho lies on the real seam; choose Im(rho) != 0.")

    circ_residue = chirre_toy_residue_circ(rho, x, sigma, T, side)
    star_residue = chirre_toy_residue_star(rho, x, sigma, T, side)

    if imag(rho) > 0:
        return circ_residue + star_residue

    return circ_residue - star_residue


def contour_shift_chirre_toy_pole(
    rho,
    x,
    sigma=0,
    T=20,
    alpha=0.5,
    side="plus",
):
    """
    Test the Chirre circ/star contour shift with a controlled toy pole.

    We take

        F(s) = 1/(s - rho),

    so the only pole is at rho.

    This lets us verify that the residue terms have the correct sign and
    normalization before using a real arithmetic function such as
    -zeta'/zeta.

    The test computes:

        original vertical integral

    and compares it against

        shifted boundary terms + 2*pi*i * residue contribution.

    Here the shifted boundary terms are assembled using the same circ/star
    decomposition as in the no-pole regularized zeta test.
    """
    if not is_inside_chirre_rectangle(rho, alpha, T):
        raise ValueError(
            "rho must lie strictly inside alpha < Re(s) < 1, "
            "-T < Im(s) < T."
        )

    if abs(imag(rho)) < 1e-12:
        raise ValueError("rho must not lie on the real seam.")

    F_toy = make_toy_pole_function(rho)

    G_circ, G_star = make_chirre_regularized_integrand_pieces(
        F_regular=F_toy,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    original = chirre_original_vertical_integral_from_pieces(G_circ, G_star, T)

    circ = chirre_circ_rectangle_integrals(G_circ, alpha, T)
    star_upper = chirre_star_upper_rectangle_integrals(G_star, alpha, T)
    star_lower = chirre_star_lower_rectangle_integrals(G_star, alpha, T)

    circ_residue = chirre_toy_residue_circ(rho, x, sigma, T, side)
    star_residue = chirre_toy_residue_star(rho, x, sigma, T, side)
    original_residue = chirre_toy_residue_for_original_integrand(
        rho=rho,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    # Circ rectangle:
    #
    #   right + top + left + bottom = 2*pi*i*circ_residue
    #
    # so
    #
    #   right = 2*pi*i*circ_residue - top - left - bottom.
    circ_shifted_right = (
        2*pi*I*circ_residue
        - circ["top"]
        - circ["left_down"]
        - circ["bottom"]
    )

    # Star rectangles depend on whether rho lies in the upper or lower half.
    #
    # Upper:
    #   right + top + left + seam = 2*pi*i*star_residue
    #
    # Lower:
    #   right + seam + left + bottom = 2*pi*i*star_residue
    #
    if imag(rho) > 0:
        star_upper_residue_term = 2*pi*I*star_residue
        star_lower_residue_term = 0
    else:
        star_upper_residue_term = 0
        star_lower_residue_term = 2*pi*I*star_residue

    star_upper_shifted_right = (
        star_upper_residue_term
        - star_upper["top"]
        - star_upper["left_down"]
        - star_upper["seam"]
    )

    star_lower_shifted_right = (
        star_lower_residue_term
        - star_lower["seam"]
        - star_lower["left_down"]
        - star_lower["bottom"]
    )

    # Original vertical:
    #
    #   circ full right + star upper right - star lower right.
    shifted = (
        circ_shifted_right
        + star_upper_shifted_right
        - star_lower_shifted_right
    )

    return {
        "rho": rho,
        "x": x,
        "sigma": sigma,
        "T": T,
        "alpha": alpha,
        "side": side,
        "original": original,
        "shifted": shifted,
        "shift_error": original - shifted,
        "circ_rectangle": circ["rectangle"],
        "star_upper_rectangle": star_upper["rectangle"],
        "star_lower_rectangle": star_lower["rectangle"],
        "circ_residue": circ_residue,
        "star_residue": star_residue,
        "original_residue": original_residue,
        "circ_rectangle_error": circ["rectangle"] - 2*pi*I*circ_residue,
        "star_upper_rectangle_error": star_upper["rectangle"] - star_upper_residue_term,
        "star_lower_rectangle_error": star_lower["rectangle"] - star_lower_residue_term,
        "original_normalized": original / (I*T),
        "shifted_normalized": shifted / (I*T),
        "shift_error_normalized": (original - shifted) / (I*T),
    }


def print_chirre_toy_pole_contour_shift_result(result):
    """
    Pretty-print the toy-pole Chirre contour-shift diagnostic.
    """
    print("=" * 70)
    print("Chirre circ/star contour shift: toy pole")
    print("=" * 70)
    print(f"rho = {N(result['rho'])}")
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"alpha = {result['alpha']}")
    print(f"side = {result['side']}")
    print()

    print("Residues:")
    print(f"  circ residue        = {N(result['circ_residue'])}")
    print(f"  star residue        = {N(result['star_residue'])}")
    print(f"  original residue    = {N(result['original_residue'])}")
    print()

    print("Rectangle checks:")
    print(f"  circ rectangle error       = {N(result['circ_rectangle_error'])}")
    print(f"  star upper rectangle error = {N(result['star_upper_rectangle_error'])}")
    print(f"  star lower rectangle error = {N(result['star_lower_rectangle_error'])}")
    print()

    print("Shift identity:")
    print(f"  original vertical    = {N(result['original'])}")
    print(f"  shifted expression   = {N(result['shifted'])}")
    print(f"  shift error          = {N(result['shift_error'])}")
    print()

    print("Normalized by iT:")
    print(f"  original vertical    = {N(result['original_normalized'])}")
    print(f"  shifted expression   = {N(result['shifted_normalized'])}")
    print(f"  shift error          = {N(result['shift_error_normalized'])}")
    print()



# ============================================================
# Chirre circ/star contour shift for -zeta'/zeta
# ============================================================




# ============================================================
# ADDED: Higher-precision zeta logarithmic derivative
# ============================================================

import mpmath as mp


def sage_complex_to_mpmath(s, dps=50):
    """
    Convert a Sage complex/real number to an mpmath complex number.

    The numerical integration points are usually only machine precision,
    so we convert through Python floats instead of asking Sage to create
    artificial high precision.
    """
    mp.mp.dps = dps

    return mp.mpc(
        float(real(s)),
        float(imag(s)),
    )


def mpmath_complex_to_sage(z):
    """
    Convert an mpmath complex number back to a Sage complex number.
    """
    return RDF(str(mp.re(z))) + I*RDF(str(mp.im(z)))


def zeta_prime_mpmath(s, dps=50):
    """
    Compute zeta'(s) using mpmath.
    """
    mp.mp.dps = dps
    smp = sage_complex_to_mpmath(s, dps=dps)

    return mpmath_complex_to_sage(mp.zeta(smp, derivative=1))


def zeta_mpmath(s, dps=50):
    """
    Compute zeta(s) using mpmath.
    """
    mp.mp.dps = dps
    smp = sage_complex_to_mpmath(s, dps=dps)

    return mpmath_complex_to_sage(mp.zeta(smp))


def minus_zeta_prime_over_zeta_regularized(s, dps=50):
    """
    Regularized logarithmic derivative

        F(s) = -zeta'(s)/zeta(s) - 1/(s - 1).

    Near s = 1,

        F(s) -> -EulerGamma.
    """
    if abs(s - 1) < 1e-10:
        return -euler_gamma

    zeta_value = zeta_mpmath(s, dps=dps)
    zeta_prime_value = zeta_prime_mpmath(s, dps=dps)

    return -zeta_prime_value / zeta_value - 1/(s - 1)







def first_zeta_zero_pair():
    """
    Return the first conjugate pair of nontrivial zeta zeros.

    These are hard-coded for this first diagnostic.
    """
    gamma1 = 14.134725141734693790457251983562
    return [
        1/2 + I*gamma1,
        1/2 - I*gamma1,
    ]


def zeros_inside_chirre_rectangle(zeros, alpha, T):
    """
    Filter a list of candidate zeros to those strictly inside

        alpha < Re(s) < 1,
        -T < Im(s) < T.
    """
    inside = []

    for rho in zeros:
        if alpha < real(rho) < 1 and -T < imag(rho) < T:
            inside.append(rho)

    return inside


def zeta_zero_residue_circ(rho, x, sigma, T, side):
    """
    Residue of the circ piece at a simple zero rho of zeta.

    Since

        -zeta'/zeta

    has residue -1 at a simple zero rho, the residue is

        -Phi_circ_lambda((rho - 1)/(iT)) x^rho.
    """
    lam = chirre_lambda(sigma, T)
    z = (rho - 1) / (I*T)

    return -chirre_Phi_circ_lambda(z, lam, side) * x**rho


def zeta_zero_residue_star(rho, x, sigma, T, side):
    """
    Residue of the star piece at a simple zero rho of zeta.

    Since

        -zeta'/zeta

    has residue -1 at rho, the residue is

        -Phi_star_lambda((rho - 1)/(iT)) x^rho.
    """
    lam = chirre_lambda(sigma, T)
    z = (rho - 1) / (I*T)

    return -chirre_Phi_star_lambda(z, lam, side) * x**rho


def zeta_zero_residue_sums(zeros, x, sigma, T, alpha, side):
    """
    Compute residue sums for the circ/star decomposition.

    Returns

        circ_residue_sum:
            sum over all zeros inside the full rectangle.

        star_upper_residue_sum:
            sum over zeros in the upper half-rectangle.

        star_lower_residue_sum:
            sum over zeros in the lower half-rectangle.

    The original piecewise integrand has residue contribution

        circ_residue_sum
        + star_upper_residue_sum
        - star_lower_residue_sum.
    """
    inside = zeros_inside_chirre_rectangle(zeros, alpha, T)

    circ_total = 0
    star_upper_total = 0
    star_lower_total = 0

    for rho in inside:
        circ_res = zeta_zero_residue_circ(rho, x, sigma, T, side)
        star_res = zeta_zero_residue_star(rho, x, sigma, T, side)

        circ_total += circ_res

        if imag(rho) > 0:
            star_upper_total += star_res
        elif imag(rho) < 0:
            star_lower_total += star_res
        else:
            raise ValueError("A zero lies on the real seam; this is not handled.")

    original_total = circ_total + star_upper_total - star_lower_total

    return {
        "zeros_inside": inside,
        "circ_residue_sum": circ_total,
        "star_upper_residue_sum": star_upper_total,
        "star_lower_residue_sum": star_lower_total,
        "original_residue_sum": original_total,
    }


def contour_shift_chirre_zeta_zero_pair(
    x,
    sigma=0,
    T=20,
    alpha=0.25,
    side="plus",
):
    """
    Test the Chirre circ/star contour shift for

        F(s) = -zeta'(s)/zeta(s) - 1/(s - 1).

    With alpha = 0.25 and T = 20, the shifted rectangle contains the
    first conjugate pair of nontrivial zeta zeros, but no trivial zeros.

    This is the first arithmetic residue test.
    """
    if not (alpha < 1/2):
        raise ValueError("For this first test, choose alpha < 1/2.")

    if T <= 14.2:
        raise ValueError("Choose T > 14.2 so the first zeta zero is inside.")

    zeros = first_zeta_zero_pair()

    residue_data = zeta_zero_residue_sums(
        zeros=zeros,
        x=x,
        sigma=sigma,
        T=T,
        alpha=alpha,
        side=side,
    )

    G_circ, G_star = make_chirre_regularized_integrand_pieces(
        F_regular=minus_zeta_prime_over_zeta_regularized,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    original = chirre_original_vertical_integral_from_pieces(G_circ, G_star, T)

    circ = chirre_circ_rectangle_integrals(G_circ, alpha, T)
    star_upper = chirre_star_upper_rectangle_integrals(G_star, alpha, T)
    star_lower = chirre_star_lower_rectangle_integrals(G_star, alpha, T)

    circ_residue_sum = residue_data["circ_residue_sum"]
    star_upper_residue_sum = residue_data["star_upper_residue_sum"]
    star_lower_residue_sum = residue_data["star_lower_residue_sum"]

    # Rectangle checks.
    circ_rectangle_error = circ["rectangle"] - 2*pi*I*circ_residue_sum

    star_upper_rectangle_error = (
        star_upper["rectangle"]
        - 2*pi*I*star_upper_residue_sum
    )

    star_lower_rectangle_error = (
        star_lower["rectangle"]
        - 2*pi*I*star_lower_residue_sum
    )

    # Shift formulas.
    circ_shifted_right = (
        2*pi*I*circ_residue_sum
        - circ["top"]
        - circ["left_down"]
        - circ["bottom"]
    )

    star_upper_shifted_right = (
        2*pi*I*star_upper_residue_sum
        - star_upper["top"]
        - star_upper["left_down"]
        - star_upper["seam"]
    )

    star_lower_shifted_right = (
        2*pi*I*star_lower_residue_sum
        - star_lower["seam"]
        - star_lower["left_down"]
        - star_lower["bottom"]
    )

    shifted = (
        circ_shifted_right
        + star_upper_shifted_right
        - star_lower_shifted_right
    )

    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "alpha": alpha,
        "side": side,
        "zeros_inside": residue_data["zeros_inside"],
        "circ_residue_sum": circ_residue_sum,
        "star_upper_residue_sum": star_upper_residue_sum,
        "star_lower_residue_sum": star_lower_residue_sum,
        "original_residue_sum": residue_data["original_residue_sum"],
        "original": original,
        "shifted": shifted,
        "shift_error": original - shifted,
        "circ_rectangle_error": circ_rectangle_error,
        "star_upper_rectangle_error": star_upper_rectangle_error,
        "star_lower_rectangle_error": star_lower_rectangle_error,
        "original_normalized": original / (I*T),
        "shifted_normalized": shifted / (I*T),
        "shift_error_normalized": (original - shifted) / (I*T),
    }


def print_chirre_zeta_zero_pair_result(result):
    """
    Pretty-print the zeta-zero residue contour-shift diagnostic.
    """
    print("=" * 70)
    print("Chirre circ/star contour shift: first zeta zero pair")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"alpha = {result['alpha']}")
    print(f"side = {result['side']}")
    print()

    print("Zeros inside:")
    for rho in result["zeros_inside"]:
        print(f"  rho = {N(rho)}")
    print()

    print("Residue sums:")
    print(f"  circ residue sum        = {N(result['circ_residue_sum'])}")
    print(f"  star upper residue sum  = {N(result['star_upper_residue_sum'])}")
    print(f"  star lower residue sum  = {N(result['star_lower_residue_sum'])}")
    print(f"  original residue sum    = {N(result['original_residue_sum'])}")
    print()

    print("Rectangle checks:")
    print(f"  circ rectangle error       = {N(result['circ_rectangle_error'])}")
    print(f"  star upper rectangle error = {N(result['star_upper_rectangle_error'])}")
    print(f"  star lower rectangle error = {N(result['star_lower_rectangle_error'])}")
    print()

    print("Shift identity:")
    print(f"  original vertical    = {N(result['original'])}")
    print(f"  shifted expression   = {N(result['shifted'])}")
    print(f"  shift error          = {N(result['shift_error'])}")
    print()

    print("Normalized by iT:")
    print(f"  original vertical    = {N(result['original_normalized'])}")
    print(f"  shifted expression   = {N(result['shifted_normalized'])}")
    print(f"  shift error          = {N(result['shift_error_normalized'])}")
    print()

