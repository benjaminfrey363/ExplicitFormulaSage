
"""
Thin wrapper to access Odlyzko zeta-zero database
"""

try:
    from sage.databases.odlyzko import zeta_zeros
except ImportError as exc:
    raise RuntimeError(
        "Could not import Sage's Odlyzko zeta-zero database module. "
        "Try running this with `sage -python`, not plain `python`."
    ) from exc

def odlyzko_zeta_zeros():
    '''
    Get zeros from database (stored as list of imaginary parts)
    '''
    try:
        return zeta_zeros()
    except Exception as exc:
        raise RuntimeError("Could not load the Odlyzko zeta-zero database.") from exc


def first_zeta_zero_imaginary_parts(n):
    '''
    Return imaginary parts of the first n zeta zeroes
    '''
    if n < 0:
        raise ValueError("n must be nonnegative")

    zeros = odlyzko_zeta_zeros()

    if n > len(zeros):
        raise ValueError(f"Requested {n} zeros, but only {len(zeros)} are available.")

    return list(zeros[:n])


def zeta_zeros_up_to_height(T):
    '''
    Return imaginary parts of zeta zeros up to height T
    '''
    if T <= 0:
        return []

    zeros = odlyzko_zeta_zeros()

    out = []
    for gamma in zeros:
        if gamma > T:
            break
        out.append(gamma)

    return out