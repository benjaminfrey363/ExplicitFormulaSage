

from sage.all import I, cot, pi






"""
Computation of CH Weight
"""

def theta_T_sigma(T, sigma):
    """
    Return the function theta_{T,sigma}(s).
    """
    if T <= 0:
        raise ValueError("T must be positive")

    def theta(s):
        return 1 - (s - sigma) / (I * T)

    return theta


def c_T_sigma(T, sigma):
    """
    Compute the normalization constant c_{T,sigma}.
    """
    theta = theta_T_sigma(T, sigma)
    theta_eval = theta(1 + I * T)

    return theta_eval * cot(pi * theta_eval)


def omega_T_sigma(T, sigma):
    """
    Return the function omega^+_{T,sigma}(s).
    """
    if T <= 0:
        raise ValueError("T must be positive")

    c = c_T_sigma(T, sigma)
    theta = theta_T_sigma(T, sigma)

    def omega(s):
        theta_eval = theta(s)
        return -theta_eval * cot(pi * theta_eval) + c

    return omega


def ch_weight_T_xi(T, xi):
    """
    Return the nonnegative CH weight

        |omega^+_{T,0}(s) + xi * theta_{T,1}(s) * i|

    as a function of s.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    omega = omega_T_sigma(T, 0)
    theta = theta_T_sigma(T, 1)

    def ch_weight(s):
        raw_weight = omega(s) + xi * theta(s) * I
        return abs(raw_weight)

    return ch_weight




"""
Lemma 7.2 Pointwise bound
"""

def F(z):
    """
    Chirre--Helfgott auxiliary function

        F(z) = 1/pi - (1 - z) cot(pi(1 - z)).
    """
    return 1 / pi - (1 - z) * cot(pi * (1 - z))


def ch_simplified_weight_T_xi(T, xi):
    """
    Return the simplified Lemma 7.2 weight

        |F(gamma/T) + xi(1 - gamma/T)i|

    as a function of s = beta + i*gamma.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    def simplified_weight(s):
        gamma = s.imag()

        if not (0 < gamma <= T):
            raise ValueError("s must satisfy 0 < Im(s) <= T")

        raw_weight = (
            F(gamma / T)
            + xi * (1 - gamma / T) * I
        )

        return abs(raw_weight)

    return simplified_weight


def ch_131_upper_bound_T_xi(T, xi):
    """
    Return the pointwise upper bound from equation (131),
    specialized to sigma = 0.

    For s = beta + i*gamma, this computes

        W_simp
        + beta*T/(pi*gamma^2)
        + (2 + 0.78*beta)/T.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if not (-1 <= xi <= 1):
        raise ValueError("xi must lie in [-1, 1]")

    simplified_weight = ch_simplified_weight_T_xi(T, xi)

    def upper_bound(s):
        beta = s.real()
        gamma = s.imag()

        if not (0 < gamma <= T):
            raise ValueError("s must satisfy 0 < Im(s) <= T")
        if not (0 <= beta <= 1):
            raise ValueError("s must lie in the critical strip")

        reciprocal_square_error = beta * T / (pi * gamma**2)
        constant_order_error = (2 + 0.78 * beta) / T

        return (
            simplified_weight(s)
            + reciprocal_square_error
            + constant_order_error
        )

    return upper_bound




"""
Playground
"""

rho = 1 / 2 + 14.1347251417347 * I
T = 100
xi = 1

exact_weight = ch_weight_T_xi(T, xi)
simplified_weight = ch_simplified_weight_T_xi(T, xi)
upper_bound_131 = ch_131_upper_bound_T_xi(T, xi)

print("Exact weight:      ", exact_weight(rho).n())
print("Simplified weight: ", simplified_weight(rho).n())
print("(131) upper bound:", upper_bound_131(rho).n())

assert exact_weight(rho).n() <= upper_bound_131(rho).n()



