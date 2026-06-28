"""
chebyshev.py

Basic implementations of the von Mangoldt function,
the Chebyshev psi function, and the Chebyshev theta function.
"""

from sage.all import log, factor, floor, primes


def von_mangoldt(n):
    """
    Return the von Mangoldt function Lambda(n).

    Lambda(n) = log(p) if n = p^k for some prime p and k >= 1,
    and 0 otherwise.

    Must be run in SageMath environment, uses the sage `factor` function
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    if n == 1:
        return 0

    fac = factor(n)

    if len(fac) == 1:
        p, _ = fac[0]
        return log(int(p))

    return 0


def chebyshev_psi(x):
    """
    Compute psi(x) = sum_{n <= x} Lambda(n),
    for x: real or integer upper bound

    Must be run in SageMath environment, uses the sage `floor` function
    """
    if x < 1:
        return 0

    return sum(von_mangoldt(n) for n in range(1, floor(x) + 1))


def chebyshev_theta(x):
    """
    Compute theta(x) = sum_{p <= x} log(p),
    for x: real or integer upper bound

    Must be run in SageMath environment, uses sage `primes` and `floor` functions
    """
    if x < 2:
        return 0

    return sum(log(int(p)) for p in primes(floor(x) + 1))