"""Inverse of a compound-symmetric matrix, Graybill's theorem."""

from ._richresult import RichResult

__all__ = ["compound_symmetry_inverse"]


def compound_symmetry_inverse(k, a, b):
    r"""Inverse of :math:`C = (a-b)I + bJ`, Graybill (1983) Theorem 8.3.4.

    Schabenberger & Gotway (2005, Theorem 1.1, p. 34): the ``k x k``
    matrix with ``a`` on the diagonal and ``b`` off it is invertible if
    and only if :math:`a \ne b` and :math:`a \ne -(k-1)b`, and then

    .. math::

        C^{-1} = \frac{1}{a-b}\left(I - \frac{b}{a + (k-1)b} J\right).

    With :math:`a = \sigma^2` and :math:`b = \rho\sigma^2` this is the
    inverse of the compound-symmetry covariance matrix, which gives
    :math:`1'\Sigma^{-1}1 = n\sigma^{-2}/\{1 + (n-1)\rho\}`.

    Parameters
    ----------
    k : int
        Dimension, at least 1.
    a : float
        Diagonal element.
    b : float
        Off-diagonal element.

    Returns
    -------
    RichResult
        ``exists`` (bool), and when it exists ``inverse`` (list of rows),
        ``diag`` and ``offdiag`` (its two distinct elements) and
        ``sum_inverse`` (:math:`1'C^{-1}1 = k/\{a + (k-1)b\}`).

    References
    ----------
    Graybill, F. A. (1983). Matrices with Applications in Statistics,
    2nd ed. Wadsworth, Theorem 8.3.4, p. 190. Schabenberger, O. &
    Gotway, C. A. (2005). Statistical Methods for Spatial Data Analysis.
    Chapman & Hall/CRC, Theorem 1.1, p. 34.
    """
    k = int(k)
    a = float(a)
    b = float(b)
    if k < 1:
        raise ValueError("`k` must be at least 1")
    exists = a != b and a != -(k - 1) * b
    payload = {"exists": exists, "k": k, "a": a, "b": b}
    if exists:
        d = a + (k - 1) * b
        off = -b / ((a - b) * d)
        diag = 1.0 / (a - b) + off
        payload.update(
            {
                "inverse": [[diag if i == j else off for j in range(k)] for i in range(k)],
                "diag": diag,
                "offdiag": off,
                "sum_inverse": k / d,
            }
        )
    return RichResult(
        title="Inverse of a compound-symmetric matrix (Graybill Thm 8.3.4)",
        summary_lines=[("k", k), ("exists", exists)],
        payload=payload,
    )


def cheatsheet():
    return "csinv: inverse of (a-b)I + bJ; exists iff a != b and a != -(k-1)b"
