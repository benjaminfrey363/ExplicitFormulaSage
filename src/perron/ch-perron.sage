"""
chirre_perron_framework.sage

Numerical framework for Chirre--Helfgott's weighted Perron-type formula.

This implements the general identity

    1/(2*pi*i*T) int_{sigma-i∞}^{sigma+i∞}
        phi(Im(s)/T) A(s) x^s ds

    =

    1/(2*pi) sum_n a_n (x/n)^sigma
        phi_hat( T/(2*pi) log(n/x) ).

When phi is supported on [-1,1], the integral is only over
sigma - iT to sigma + iT.

This is the level of generality of Chirre--Helfgott Lemma 2.1.
"""


# ============================================================
# General numerical Fourier transform
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
        Integration interval, usually (-1, 1).

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
    Compute the integral side of Chirre--Helfgott's weighted Perron formula.

    We compute

        1/(2*pi*i*T) int_{sigma-iT}^{sigma+iT}
            phi(Im(s)/T) A(s) x^s ds.

    Parametrize s = sigma + i*T*u, where u in [-1,1].
    Then ds = i*T du, so this becomes

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


def compare_weighted_perron(A, a, phi, phi_hat, x, sigma, T, N_max):
    """
    Compare integral side and truncated sum side.

    This is mainly a diagnostic function.
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


# ============================================================
# Simple test weight: phi = 1_{[-1,1]}
# ============================================================

def box_phi(t):
    """
    The simple compactly supported weight phi(t) = 1 on [-1,1].

    Since we only integrate over [-1,1], this just returns 1.
    """
    return 1


def box_phi_hat(y):
    """
    Fourier transform of 1_{[-1,1]}.

        int_{-1}^{1} exp(-2*pi*i*y*t) dt
        = sin(2*pi*y)/(pi*y),

    with limiting value 2 at y = 0.
    """
    if abs(y) < 1e-12:
        return 2

    return sin(2*pi*y)/(pi*y)


# ============================================================
# Dirichlet series test case: A(s) = zeta(s)
# ============================================================

def A_zeta(s):
    """
    A(s) = zeta(s) = sum 1/n^s.
    """
    return zeta(s)


def a_zeta(n):
    """
    Coefficients of zeta(s): a_n = 1.
    """
    return 1


def test_weighted_perron_with_zeta_box():
    """
    Test the weighted Perron identity using A(s)=zeta(s)
    and phi(t)=1_{[-1,1]}.

    We choose sigma > 1 so that the Dirichlet series converges absolutely
    on Re(s)=sigma.
    """
    x = 25.7
    sigma = 2
    T = 20
    N_max = 5000

    result = compare_weighted_perron(
        A=A_zeta,
        a=a_zeta,
        phi=box_phi,
        phi_hat=box_phi_hat,
        x=x,
        sigma=sigma,
        T=T,
        N_max=N_max,
    )

    print("=" * 70)
    print("Weighted Perron test: A(s)=zeta(s), phi=box")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"N_max = {result['N_max']}")
    print()
    print(f"integral side       = {N(result['integral_side'])}")
    print(f"truncated sum side  = {N(result['sum_side'])}")
    print(f"difference          = {N(result['difference'])}")
    print(f"absolute difference = {N(result['absolute_difference'])}")
    print()


# ============================================================
# Chirre--Helfgott truncated exponential I_lambda
# ============================================================

def real_sign(x):
    """
    Sign function returning -1, 0, or 1.
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def chirre_lambda(sigma, T):
    """
    Chirre--Helfgott parameter

        lambda = 2*pi*(sigma - 1)/T.
    """
    return 2*pi*(sigma - 1)/T


def truncated_exponential_I_lambda(y, lam):
    """
    Chirre--Helfgott truncated exponential

        I_lambda(y) = 1_{[0,∞)}(sgn(lambda) y) exp(-lambda y).

    This is the sharp cutoff object that the majorant/minorant approximate.
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    if real_sign(lam) * y >= 0:
        return exp(-lam*y)

    return 0


# ============================================================
# Chirre--Helfgott / Graham--Vaaler compactly supported phi
# ============================================================

def Phi_circ(nu, z, side):
    """
    Phi^{±,circ}_nu(z).

    side should be either "plus" or "minus".
    """
    if side == "plus":
        eps = 1
    elif side == "minus":
        eps = -1
    else:
        raise ValueError("side must be 'plus' or 'minus'")

    w = -2*pi*I*z + nu
    return (1/2) * (coth(w/2) + eps)


def Phi_star(nu, z, side):
    """
    Phi^{±,star}_nu(z).

    side should be either "plus" or "minus".
    """
    if side == "plus":
        eps = 1
    elif side == "minus":
        eps = -1
    else:
        raise ValueError("side must be 'plus' or 'minus'")

    w = -2*pi*I*z + nu

    return (I/(2*pi)) * (
        (nu/2)*coth(nu/2)
        - (w/2)*coth(w/2)
        + eps*pi*I*z
    )


def chirre_phi_nu(t, nu, side):
    """
    Compactly supported Chirre--Helfgott/Graham--Vaaler function

        phi^{±}_nu(t)

    supported on [-1,1].

    The formula is

        phi^{±}_nu(t)
        =
        1_{[-1,1]}(t)
        ( Phi^{±,circ}_nu(t) + sgn(t) Phi^{±,star}_nu(t) ).
    """
    if abs(t) > 1:
        return 0

    return Phi_circ(nu, t, side) + real_sign(t)*Phi_star(nu, t, side)


def chirre_phi_lambda(t, lam, side):
    """
    Chirre--Helfgott function phi^{±}_lambda(t).

    They define

        phi^{±}_lambda(t)
        =
        phi^{±}_{|lambda|}(sgn(lambda) t).
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero")

    nu = abs(lam)
    return chirre_phi_nu(real_sign(lam)*t, nu, side)


def make_chirre_phi(lam, side):
    """
    Return a one-variable function t |-> phi^{±}_lambda(t).
    """
    return lambda t: chirre_phi_lambda(t, lam, side)


def make_chirre_phi_hat_numeric(lam, side):
    """
    Return a numerical Fourier transform function for phi^{±}_lambda.

    This is slower than a closed-form expression, but it is useful for
    first experiments.
    """
    phi = make_chirre_phi(lam, side)

    return lambda y: fourier_transform_compact(phi, y, support=(-1, 1))


def test_chirre_weight_shape():
    """
    Print a few values comparing I_lambda and the numerical Fourier
    transforms of the Chirre majorant/minorant.

    This is a first sanity check. It is not a proof.
    """
    sigma = 2
    T = 20
    lam = chirre_lambda(sigma, T)

    phi_plus_hat = make_chirre_phi_hat_numeric(lam, "plus")
    phi_minus_hat = make_chirre_phi_hat_numeric(lam, "minus")

    y_values = [-3, -1, -0.25, 0, 0.25, 1, 3]

    print("=" * 70)
    print("Chirre weight sanity check")
    print("=" * 70)
    print(f"sigma = {sigma}")
    print(f"T = {T}")
    print(f"lambda = {N(lam)}")
    print()
    print("y        minorant_hat        I_lambda          majorant_hat")
    print("-" * 70)

    for y in y_values:
        lower = phi_minus_hat(y)
        target = truncated_exponential_I_lambda(y, lam)
        upper = phi_plus_hat(y)

        print(f"{y:<8} {N(real(lower)):<18} {N(target):<18} {N(real(upper))}")




# Test for worst inequality failure (hopefully some kind of floating point error)
# SLOW
def test_chirre_inequality_grid():
    sigma = 2
    T = 20
    lam = chirre_lambda(sigma, T)

    phi_plus_hat = make_chirre_phi_hat_numeric(lam, "plus")
    phi_minus_hat = make_chirre_phi_hat_numeric(lam, "minus")

    #y_values = [QQ(k)/10 for k in range(-50, 51)]
    # Much smaller sample
    y_values = [-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3]

    worst_lower_violation = 0
    worst_upper_violation = 0

    for y in y_values:
        lower = real(phi_minus_hat(y))
        target = truncated_exponential_I_lambda(y, lam)
        upper = real(phi_plus_hat(y))

        lower_violation = max(0, lower - target)
        upper_violation = max(0, target - upper)

        worst_lower_violation = max(worst_lower_violation, lower_violation)
        worst_upper_violation = max(worst_upper_violation, upper_violation)

    print("Worst lower violation:", N(worst_lower_violation))
    print("Worst upper violation:", N(worst_upper_violation))




# ============================================================
# Main script
# ============================================================

test_weighted_perron_with_zeta_box()
test_chirre_weight_shape()


# slow test
test_chirre_inequality_grid()