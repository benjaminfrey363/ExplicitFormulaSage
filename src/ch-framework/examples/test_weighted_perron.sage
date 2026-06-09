"""
test_weighted_perron.sage

Basic diagnostic test for weighted_perron.sage using A(s)=zeta(s)
and phi(t)=1 on [-1,1].
"""

load("src/ch-framework/weighted_perron.sage")


# ============================================================
# Simple test weight: phi = 1_{[-1,1]}
# ============================================================

def box_phi(t):
    return 1


def box_phi_hat(y):
    """
    Fourier transform of 1_{[-1,1]}:

        int_{-1}^{1} exp(-2*pi*i*y*t) dt
        = sin(2*pi*y)/(pi*y).

    The limiting value at y=0 is 2.
    """
    if abs(y) < 1e-12:
        return 2

    return sin(2*pi*y)/(pi*y)


# ============================================================
# Test Dirichlet series: A(s)=zeta(s)
# ============================================================

def A_zeta(s):
    return zeta(s)


def a_zeta(n):
    return 1


def test_weighted_perron_with_zeta_box():
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


test_weighted_perron_with_zeta_box()