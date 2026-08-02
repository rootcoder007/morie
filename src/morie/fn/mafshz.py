# morie.fn -- function file (rootcoder007/morie)
"""Fisher's z transformation of a correlation (Fisher 1921)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_fishers_z"]


def ma_fishers_z(r, n=None):
    r"""Variance-stabilising transform of a Pearson correlation.

    .. math::

        z = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right) = \operatorname{arctanh}(r),
        \qquad \operatorname{Var}(z) \approx \frac{1}{n-3}

    Parameters
    ----------
    r : float or array-like
        Correlation(s), strictly inside :math:`(-1, 1)`.
    n : int, optional
        Sample size the correlation was computed from. Required for the
        variance; must satisfy :math:`n > 3`.

    Returns
    -------
    RichResult
        keys: ``z``, ``var`` (``None`` when ``n`` is not given), ``se``,
        ``r``, ``n``, ``method``.

    Raises
    ------
    ValueError
        If :math:`|r| \ge 1`, or if ``n <= 3``.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of
        correlation deduced from a small sample. *Metron*, 1, 3-32.

    Notes
    -----
    The transform exists because the sampling distribution of :math:`r` is
    sharply skewed near :math:`\pm 1`, so averaging raw correlations across
    studies is biased toward zero. :math:`z` is approximately normal with a
    variance that does not depend on :math:`\rho`, which is what makes
    inverse-variance weighting valid in meta-analysis.

    :math:`|r| = 1` is excluded rather than returned as :math:`\pm\infty`:
    the variance :math:`1/(n-3)` is a large-sample approximation that carries
    no meaning at the boundary.

    :math:`n > 3` is required, not :math:`n \ge 3`. At :math:`n = 3` the
    variance is a division by zero and at :math:`n < 3` it is *negative*,
    which would propagate a nonsensical standard error into every downstream
    weight.
    """
    rr = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(rr)):
        raise ValueError(f"r must be finite; got {r!r}")
    if np.any(np.abs(rr) >= 1.0):
        raise ValueError(
            f"Fisher's z requires |r| < 1; got {r!r}. At |r| = 1 the transform "
            "diverges and the 1/(n-3) variance approximation does not apply."
        )
    z = np.arctanh(rr)
    var = None
    se = None
    if n is not None:
        n = int(n)
        if n <= 3:
            raise ValueError(
                f"n must be > 3 for Var(z) = 1/(n-3); got {n}. At n = 3 this is a "
                "division by zero and below it the variance would be negative."
            )
        var = 1.0 / (n - 3)
        se = float(np.sqrt(var))
    scalar = z.ndim == 0
    return RichResult(
        payload={
            "z": float(z) if scalar else z,
            "var": var,
            "se": se,
            "r": float(rr) if scalar else rr,
            "n": n,
            "method": "Fisher's z transform, Var(z) = 1/(n-3) (Fisher 1921)",
        }
    )


def cheatsheet():
    return "mafshz: z = arctanh(r) = 0.5 ln((1+r)/(1-r)), Var = 1/(n-3) (Fisher 1921)."
