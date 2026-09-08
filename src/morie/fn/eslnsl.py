# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Natural cubic spline basis (ESL Ch 5.2.1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_natural_spline"]


def esl_natural_spline(x, knots):
    """
    Natural cubic spline basis, linear beyond the boundary knots.

    A cubic spline with K knots has K + 4 free parameters; a NATURAL
    spline adds the constraint that the function is linear beyond the
    boundary knots, which removes four of them and leaves K. That
    constraint exists because polynomial fits behave worst exactly
    where the data run out, and ESL Ch 5.2.1 notes the price is bias
    near the boundary in exchange for far smaller variance there.

    Uses ESL's Eq. 5.4-5.5 construction: N_1 = 1, N_2 = x, and
    N_{k+2} = d_k(x) - d_{K-1}(x), where
    d_k(x) = [(x-xi_k)_+^3 - (x-xi_K)_+^3] / (xi_K - xi_k).

    Parameters
    ----------
    x : array-like
        Evaluation points.
    knots : array-like
        At least 3 distinct knots (the first and last are boundaries).

    Returns
    -------
    result : dict
        Keys: estimate (number of basis functions), basis (row-major
        n x K), n_basis, knots, n, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 5.2.1 (Eq. 5.4-5.5).

    Examples
    --------
    K knots give exactly K basis functions, against K + 4 for an
    unconstrained cubic spline:

    >>> out = esl_natural_spline([0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 3.0])
    >>> out["n_basis"]
    3
    >>> out["basis"][:3]            # row for x = 0: N1 = 1, N2 = x
    [1.0, 0.0, 0.0]

    Linearity beyond the boundary is the defining property — the
    third basis function stops curving once past the last knot:

    >>> far = esl_natural_spline([3.0, 4.0, 5.0], [0.0, 1.5, 3.0])
    >>> b = far["basis"]
    >>> d1, d2 = b[5] - b[2], b[8] - b[5]
    >>> abs(d1 - d2) < 1e-9
    True
    >>> esl_natural_spline([0.0], [0.0, 1.0])
    Traceback (most recent call last):
        ...
    ValueError: a natural spline needs at least 3 distinct knots; got 2.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    kn = np.unique(np.atleast_1d(np.asarray(knots, dtype=float)))
    K = kn.size
    if K < 3:
        raise ValueError(f"a natural spline needs at least 3 distinct knots; got {K}.")
    xK, xKm1 = kn[-1], kn[-2]

    def d(k):
        return ((np.where(x > kn[k], (x - kn[k]) ** 3, 0.0)
                 - np.where(x > xK, (x - xK) ** 3, 0.0)) / (xK - kn[k]))

    cols = [np.ones_like(x), x]
    dKm1 = ((np.where(x > xKm1, (x - xKm1) ** 3, 0.0)
             - np.where(x > xK, (x - xK) ** 3, 0.0)) / (xK - xKm1))
    for k in range(K - 2):
        cols.append(d(k) - dKm1)
    B = np.column_stack(cols)
    return RichResult(payload={
        "estimate": int(B.shape[1]),
        "basis": [float(v) for v in B.ravel()],
        "n_basis": int(B.shape[1]),
        "knots": [float(v) for v in kn], "n": int(x.size),
        "method": "natural cubic spline (ESL Eq. 5.4-5.5); linear past the boundary knots"})


def cheatsheet():
    return "eslnsl: K knots -> K basis fns; linear beyond boundaries trades bias for variance"
