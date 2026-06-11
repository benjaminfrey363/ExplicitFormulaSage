"""
explicit_formula_chirre.sage

First explicit-formula-style packaging of the Chirre--Helfgott contour
shift for

    A(s) = -zeta'(s)/zeta(s).

This file assumes the lower-level files have already been loaded:

    weighted_perron.sage
    chirre_weights.sage
    smoothed_bounds.sage
    integral_bounds.sage
    contour_shift.sage

The goal here is not yet to prove a theorem-level explicit bound. Instead,
we package the numerical contour shift into named explicit-formula pieces:

    main pole term
    zero residue contribution
    shifted contour contribution
    seam/star contribution
    exact Chebyshev comparison

For now we work with the same circ/star decomposition already tested in
contour_shift.sage.
"""


# ============================================================
# Von Mangoldt / Chebyshev helpers
# ============================================================

def von_mangoldt_basic(n):
    """
    Basic von Mangoldt function.

    Returns log(p) if n = p^k for some prime p and k >= 1.
    Returns 0 otherwise.

    This duplicates a little functionality so this file can be tested
    independently of src/primecounting/chebyshev.py.
    """
    if n < 1:
        raise ValueError("n must be positive")

    fac = factor(n)

    if len(fac) == 1:
        p, k = fac[0]
        return log(p)

    return 0


def chebyshev_psi_basic(x):
    """
    Compute

        psi(x) = sum_{n <= x} Lambda(n)

    directly.
    """
    if x < 1:
        return 0

    total = 0

    for n in range(1, floor(x) + 1):
        total += von_mangoldt_basic(n)

    return total


# ============================================================
# Smoothed sum for Lambda(n)
# ============================================================

def a_von_mangoldt(n):
    """
    Coefficients of -zeta'/zeta:

        a_n = Lambda(n).
    """
    return von_mangoldt_basic(n)


def chirre_smoothed_psi_sum(x, sigma, T, side, N_max):
    """
    Compute the Chirre smoothed sum for Lambda(n):

        x^(-sigma) sum_{n <= N_max}
            Lambda(n) * x/n *
            hat(phi_lambda^side)(T/(2*pi) log(n/x)).

    This is the sum-side smoothed quantity corresponding to the
    lower/upper Chirre weights.
    """
    return chirre_smoothed_sum(
        a=a_von_mangoldt,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
        N_max=N_max,
    )


# ============================================================
# Pole term for -zeta'/zeta
# ============================================================

def chirre_zeta_log_derivative_pole_term(x, sigma, T, side):
    """
    Main pole term coming from the pole of -zeta'/zeta at s = 1.

    Since

        -zeta'/zeta(s)

    has residue 1 at s=1, the pole contribution is the same structural
    term as in integral_bounds.sage:

        2*pi*x^(1-sigma)/T * phi_lambda^side(0).

    This is the main x-term in the explicit formula.
    """
    return pole_main_term(
        sigma=sigma,
        T=T,
        x=x,
        side=side,
    )


def chirre_sigma_cutoff_correction(sigma):
    """
    Same cutoff correction as in integral_bounds.sage:

        1_{sigma < 1}/(sigma - 1).

    This appears in the normalized integral-side bound.
    """
    return sharp_cutoff_tail_correction(sigma)


# ============================================================
# Zero residue contribution
# ============================================================

def chirre_zero_residue_contribution(x, sigma, T, alpha, side, max_zeros=None):
    """
    Compute the combined zero residue contribution for

        -zeta'/zeta

    using all zeros with |Im rho| < T available from the Odlyzko database.

    This returns the residue sum before multiplying by 2*pi*i.
    """
    zeros = zeta_zero_pairs_up_to_height(
        T=T,
        max_zeros=max_zeros,
    )

    return zeta_zero_residue_sums(
        zeros=zeros,
        x=x,
        sigma=sigma,
        T=T,
        alpha=alpha,
        side=side,
    )


# ============================================================
# Shifted contour pieces
# ============================================================

def chirre_zeta_shifted_contour_pieces(x, sigma, T, alpha, side):
    """
    Compute the shifted contour pieces for

        F_regular(s) = -zeta'/zeta(s) - 1/(s-1).

    This is the regularized part after removing the pole at s=1.

    Returns the circ, star_upper, and star_lower contour dictionaries
    already used in contour_shift.sage.
    """
    G_circ, G_star = make_chirre_regularized_integrand_pieces(
        F_regular=minus_zeta_prime_over_zeta_regularized,
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    original = chirre_original_vertical_integral_from_pieces(
        G_circ,
        G_star,
        T,
    )

    circ = chirre_circ_rectangle_integrals(G_circ, alpha, T)
    star_upper = chirre_star_upper_rectangle_integrals(G_star, alpha, T)
    star_lower = chirre_star_lower_rectangle_integrals(G_star, alpha, T)

    return {
        "original": original,
        "circ": circ,
        "star_upper": star_upper,
        "star_lower": star_lower,
    }


def chirre_shifted_boundary_expression(contour_data, residue_data):
    """
    Assemble the shifted expression for the regularized vertical integral.

    This mirrors the already-tested formula in contour_shift.sage:

        circ right
        + star upper right
        - star lower right.

    Each right side is expressed using residue terms minus the remaining
    shifted boundary pieces.
    """
    circ = contour_data["circ"]
    star_upper = contour_data["star_upper"]
    star_lower = contour_data["star_lower"]

    circ_residue_sum = residue_data["circ_residue_sum"]
    star_upper_residue_sum = residue_data["star_upper_residue_sum"]
    star_lower_residue_sum = residue_data["star_lower_residue_sum"]

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
        "circ_shifted_right": circ_shifted_right,
        "star_upper_shifted_right": star_upper_shifted_right,
        "star_lower_shifted_right": star_lower_shifted_right,
        "shifted": shifted,
    }


# ============================================================
# Full diagnostic decomposition
# ============================================================

def chirre_explicit_formula_diagnostic(
    x,
    sigma=0,
    T=45,
    alpha=0.25,
    side="plus",
    N_max=None,
    max_zeros=None,
):
    """
    Compute a first explicit-formula-style diagnostic for psi(x).

    The output compares four objects:

    1. exact psi(x),
    2. sum-side smoothed psi,
    3. integral-side expression from the vertical integral,
    4. shifted-contour expression using zero residues and boundary terms.

    Parameters
    ----------
    x : real
        Evaluation point.
    sigma : real
        Chirre sigma parameter, usually sigma < 1 for psi(x)-type sums.
    T : real
        Height.
    alpha : real
        Left contour real part.
    side : str
        'plus' or 'minus'.
    N_max : int or None
        Truncation for smoothed sum. If None, uses max(ceil(10*x), 100).
    max_zeros : int or None
        Optional zero cap for debugging.
    """
    if side not in {"plus", "minus"}:
        raise ValueError("side must be 'plus' or 'minus'")

    if sigma == 1:
        raise ValueError("sigma must not equal 1")

    if alpha >= 1/2:
        raise ValueError("For zero residues, choose alpha < 1/2.")

    if N_max is None:
        N_max = max(ceil(10*x), 100)

    exact_psi = chebyshev_psi_basic(x)

    smoothed_sum = chirre_smoothed_psi_sum(
        x=x,
        sigma=sigma,
        T=T,
        side=side,
        N_max=N_max,
    )

    main_pole = chirre_zeta_log_derivative_pole_term(
        x=x,
        sigma=sigma,
        T=T,
        side=side,
    )

    cutoff_correction = chirre_sigma_cutoff_correction(sigma)

    residue_data = chirre_zero_residue_contribution(
        x=x,
        sigma=sigma,
        T=T,
        alpha=alpha,
        side=side,
        max_zeros=max_zeros,
    )

    contour_data = chirre_zeta_shifted_contour_pieces(
        x=x,
        sigma=sigma,
        T=T,
        alpha=alpha,
        side=side,
    )

    shifted_data = chirre_shifted_boundary_expression(
        contour_data=contour_data,
        residue_data=residue_data,
    )

    regularized_vertical_normalized = (
        x**(-sigma) * contour_data["original"] / (I*T)
    )

    shifted_regularized_normalized = (
        x**(-sigma) * shifted_data["shifted"] / (I*T)
    )

    # Integral-side expression:
    #
    #   main pole
    #   + normalized regularized vertical integral
    #   + cutoff correction.
    #
    # This should match the smoothed sum side, up to numerical/truncation
    # errors.
    integral_side = (
        main_pole
        + regularized_vertical_normalized
        + cutoff_correction
    )

    shifted_integral_side = (
        main_pole
        + shifted_regularized_normalized
        + cutoff_correction
    )

    zero_residue_normalized = (
        x**(-sigma)
        * 2*pi*I*residue_data["original_residue_sum"]
        / (I*T)
    )

    boundary_normalized = shifted_regularized_normalized - zero_residue_normalized


    implied_pole_term = (
        smoothed_sum
        - regularized_vertical_normalized
        - cutoff_correction
    )

    pole_term_discrepancy = main_pole - implied_pole_term


    return {
        "x": x,
        "sigma": sigma,
        "T": T,
        "alpha": alpha,
        "side": side,
        "N_max": N_max,
        "max_zeros": max_zeros,
        "num_zero_pairs": len(residue_data["zeros_inside"]) // 2,

        "exact_psi": exact_psi,
        "smoothed_sum": smoothed_sum,
        "main_pole": main_pole,
        "cutoff_correction": cutoff_correction,

        "regularized_vertical_normalized": regularized_vertical_normalized,
        "shifted_regularized_normalized": shifted_regularized_normalized,
        "zero_residue_normalized": zero_residue_normalized,
        "boundary_normalized": boundary_normalized,

        "integral_side": integral_side,
        "shifted_integral_side": shifted_integral_side,

        "sum_minus_integral": smoothed_sum - integral_side,
        "integral_minus_shifted": integral_side - shifted_integral_side,
        "exact_minus_smoothed_real": exact_psi - real(smoothed_sum),

        "residue_data": residue_data,
        "contour_data": contour_data,
        "shifted_data": shifted_data,

        "implied_pole_term": implied_pole_term,
        "pole_term_discrepancy": pole_term_discrepancy,
    }


def print_chirre_explicit_formula_diagnostic(result):
    """
    Pretty-print the explicit-formula diagnostic.
    """
    print("=" * 70)
    print("Chirre explicit-formula diagnostic for psi(x)")
    print("=" * 70)
    print(f"x = {result['x']}")
    print(f"sigma = {result['sigma']}")
    print(f"T = {result['T']}")
    print(f"alpha = {result['alpha']}")
    print(f"side = {result['side']}")
    print(f"N_max = {result['N_max']}")
    print(f"max_zeros = {result['max_zeros']}")
    print(f"number of zero pairs = {result['num_zero_pairs']}")
    print()

    print("Main quantities:")
    print(f"  exact psi(x)              = {N(result['exact_psi'])}")
    print(f"  smoothed sum              = {N(result['smoothed_sum'])}")
    print(f"  integral side             = {N(result['integral_side'])}")
    print(f"  shifted integral side     = {N(result['shifted_integral_side'])}")
    print()

    print("Decomposition:")
    print(f"  main pole                 = {N(result['main_pole'])}")
    print(f"  zero residues normalized  = {N(result['zero_residue_normalized'])}")
    print(f"  boundary normalized       = {N(result['boundary_normalized'])}")
    print(f"  cutoff correction         = {N(result['cutoff_correction'])}")
    print(f"  implied pole term        = {N(result['implied_pole_term'])}")
    print(f"  pole term discrepancy    = {N(result['pole_term_discrepancy'])}")
    print()

    print("Consistency checks:")
    print(f"  smoothed - integral       = {N(result['sum_minus_integral'])}")
    print(f"  integral - shifted        = {N(result['integral_minus_shifted'])}")
    print(f"  exact psi - Re(smoothed)  = {N(result['exact_minus_smoothed_real'])}")
    print()