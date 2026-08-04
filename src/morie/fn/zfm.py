"""Z-transform of a discrete-time sequence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["z_transform"]


def z_transform(x, z, n0=0):
    """
    Z-transform

    Formula: X(z) = sum_n x_n z ** (-n)

    The two-sided z-transform of the sequence ``x``, evaluated at the
    complex point or points ``z``.  ``n0`` is the time index of the first
    sample, so the sum runs over ``n = n0, ..., n0 + N - 1``; ``n0 = 0``
    gives the causal finite-length case
    ``X(z) = sum_{n=0}^{N-1} x_n z ** (-n)``, which is the transfer
    function of an FIR system.

    A finite-length sequence converges everywhere except possibly at
    ``z = 0`` (when the sequence has samples at ``n > 0``, which put a
    pole there) and at infinity, so no region of convergence has to be
    supplied.

    Also reported is the pole-free stability check implied by Jury's
    criterion for the causal case: an FIR sequence has all its poles at
    the origin, so it is always stable.  For that reason only the
    coefficients, the evaluation points and the values are returned; a
    recursive-filter stability test needs the denominator polynomial,
    which this function does not take.

    Parameters
    ----------
    x : array-like
        Samples ``x(n0), x(n0 + 1), ..., x(n0 + N - 1)``.
    z : complex or sequence of complex
        Point or points at which to evaluate the transform.
    n0 : int, optional
        Time index of the first sample.  Default 0 (causal).

    Returns
    -------
    result : RichResult
        Keys: X, z, coefficients, n, causal, degree, method.

    See Also
    --------
    morie.fn.bsaxfrm.ztrans : the same transform, which this function
        delegates to; it is the canonical implementation in morie.

    References
    ----------
    Jury E I (1964).  Theory and Application of the z-Transform Method.
    Wiley, New York.  The definition above is the one in Chapter 1.
    """
    from .bsaxfrm import ztrans

    if z is None:
        raise ValueError("z must be given; use bsaxfrm.ztrans for the "
                         "coefficients-only form")
    return ztrans(x, z=z, n0=n0)


def cheatsheet():
    return "zfm: z-transform X(z) = sum x_n z^-n (Jury 1964)"


# compact alias per ledger/NAMING.md
ztransform = z_transform
