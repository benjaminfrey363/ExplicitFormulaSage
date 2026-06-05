"""
Utilities for accessing Riemann zeta zeros.

We first try Sage's optional Odlyzko zeta-zero database. If it is not
available, we fall back to a small built-in list of the first few
imaginary parts of nontrivial zeta zeros.
"""


# First few positive imaginary parts gamma_n of zeros rho_n = 1/2 + i gamma_n.
# These are enough for demos and tests.
_BUILTIN_ZETA_ZERO_GAMMAS = [
    14.134725141734693790,
    21.022039638771554993,
    25.010857580145688763,
    30.424876125859513210,
    32.935061587739189691,
    37.586178158825671257,
    40.918719012147495187,
    43.327073280914999519,
    48.005150881167159728,
    49.773832477672302181,
    52.970321477714460644,
    56.446247697063394804,
    59.347044002602353079,
    60.831778524609809844,
    65.112544048081606660,
    67.079810529494173714,
    69.546401711173979252,
    72.067157674481907582,
    75.704690699083933168,
    77.144840068874805373,
]


def _try_sage_odlyzko_zeros():
    """
    Try to return Sage's Odlyzko zeta-zero database.

    Returns
    -------
    object or None
        The Sage zeta-zero list if available, otherwise None.
    """
    try:
        zeros = zeta_zeros()
        # Force access to detect missing database files.
        len(zeros)
        return zeros
    except Exception:
        return None


def first_zeta_zero_imaginary_parts(n, source="auto"):
    """
    Return the imaginary parts of the first n nontrivial zeta zeros.

    Parameters
    ----------
    n : int
        Number of zeros to return.
    source : str
        One of "auto", "sage", or "builtin".

        - "auto": use Sage Odlyzko database if available, otherwise builtin.
        - "sage": require Sage Odlyzko database.
        - "builtin": use small built-in list.

    Returns
    -------
    list
        Imaginary parts gamma_1, ..., gamma_n.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")

    if source not in {"auto", "sage", "builtin"}:
        raise ValueError("source must be one of 'auto', 'sage', or 'builtin'")

    if source in {"auto", "sage"}:
        zeros = _try_sage_odlyzko_zeros()

        if zeros is not None:
            if n > len(zeros):
                raise ValueError(
                    f"Requested {n} zeros, but Sage database only has {len(zeros)}."
                )
            return list(zeros[:n])

        if source == "sage":
            raise RuntimeError(
                "Sage's Odlyzko zeta-zero database is not available in this "
                "Sage installation. Try source='builtin' for demos, or install "
                "a Sage distribution that includes optional database packages."
            )

    # Built-in fallback.
    if n > len(_BUILTIN_ZETA_ZERO_GAMMAS):
        raise ValueError(
            f"Built-in fallback only contains {len(_BUILTIN_ZETA_ZERO_GAMMAS)} zeros. "
            "Use source='sage' with an installation that has the Odlyzko database, "
            "or add support for loading zeros from a local file."
        )

    return _BUILTIN_ZETA_ZERO_GAMMAS[:n]


def zeta_zeros_up_to_height(T, source="auto"):
    """
    Return imaginary parts of zeta zeros with 0 < gamma <= T.

    Parameters
    ----------
    T : float
        Height cutoff.
    source : str
        One of "auto", "sage", or "builtin".
    """
    if T <= 0:
        return []

    if source in {"auto", "sage"}:
        zeros = _try_sage_odlyzko_zeros()

        if zeros is not None:
            return [gamma for gamma in zeros if gamma <= T]

        if source == "sage":
            raise RuntimeError(
                "Sage's Odlyzko zeta-zero database is not available."
            )

    return [gamma for gamma in _BUILTIN_ZETA_ZERO_GAMMAS if gamma <= T]