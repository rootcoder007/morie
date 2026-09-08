# morie.fn -- function file (rootcoder007/morie)
"""Aitchison isometric log-ratio balance."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["aitchison_balance"]


def aitchison_balance(x, numerator, denominator):
    r"""An isometric log-ratio balance between two groups of parts.

    .. math::
        b = \sqrt{\frac{rs}{r+s}}\,
            \log \frac{g(x_{\mathcal{N}})}{g(x_{\mathcal{D}})},

    with :math:`g` the geometric mean and :math:`r, s` the group sizes.

    Compositional data carries only **relative** information: proportions sum
    to a constant, so the parts are not free to vary independently and
    ordinary statistics on them are meaningless. The classic symptom is
    spurious negative correlation -- if one part rises the others must fall,
    whatever the underlying process. A balance sidesteps this by measuring one
    group of parts against another on the log scale, where the constraint
    disappears.

    Balances are scale-invariant: multiplying the whole composition by any
    constant leaves them unchanged, which is what makes them comparable across
    samples measured in different totals. The :math:`\sqrt{rs/(r+s)}` factor
    makes the balance an isometry, so Euclidean distance between balance
    vectors equals Aitchison distance between compositions.

    Zeros are fatal -- the log of zero is undefined and no amount of care
    downstream repairs it. Zero replacement must happen before this is called.

    Parameters
    ----------
    x : array-like
        Composition ``(p,)`` or ``(n, p)``, strictly positive.
    numerator, denominator : sequence of int
        Disjoint index groups.

    Returns
    -------
    RichResult
        ``balance``, ``normalizer``, ``geometric_mean_num``,
        ``geometric_mean_den``.

    References
    ----------
    Egozcue, J. J., & Pawlowsky-Glahn, V. (2005). Groups of parts and their
        balances in compositional data analysis. *Mathematical Geology*,
        37(7), 795-828.

    Examples
    --------
    A balance is invariant to the total, which is what makes compositions
    comparable.

    >>> import numpy as np
    >>> c = np.array([20.0, 30.0, 50.0])
    >>> a = aitchison_balance(c, [0], [1, 2])["balance"]
    >>> b = aitchison_balance(c * 7.3, [0], [1, 2])["balance"]
    >>> bool(abs(float(a) - float(b)) < 1e-12)
    True

    Shifting mass from the denominator group to the numerator raises it.

    >>> up = aitchison_balance([40.0, 25.0, 35.0], [0], [1, 2])["balance"]
    >>> bool(float(up) > float(a))
    True

    Zeros are refused rather than producing silent infinities.

    >>> aitchison_balance([1.0, 0.0, 2.0], [0], [1, 2])
    Traceback (most recent call last):
        ...
    ValueError: compositions must be strictly positive; replace zeros first
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if np.any(X <= 0):
        raise ValueError("compositions must be strictly positive; replace zeros first")
    num = np.atleast_1d(np.asarray(numerator, dtype=int)).ravel()
    den = np.atleast_1d(np.asarray(denominator, dtype=int)).ravel()
    if np.intersect1d(num, den).size:
        raise ValueError("numerator and denominator groups must be disjoint")
    p = X.shape[1]
    if np.any(num >= p) or np.any(den >= p) or np.any(num < 0) or np.any(den < 0):
        raise ValueError(f"group indices must lie in 0..{p - 1}")
    r, s = num.size, den.size
    if r == 0 or s == 0:
        raise ValueError("both groups must be non-empty")
    gn = np.exp(np.log(X[:, num]).mean(axis=1))
    gd = np.exp(np.log(X[:, den]).mean(axis=1))
    norm = np.sqrt(r * s / (r + s))
    bal = norm * np.log(gn / gd)
    return RichResult(
        title="Aitchison balance",
        summary_lines=[("parts", int(p)), ("r", int(r)), ("s", int(s)),
                       ("normalizer", float(norm))],
        payload={
            "balance": bal if bal.size > 1 else float(bal[0]),
            "normalizer": float(norm),
            "geometric_mean_num": gn if gn.size > 1 else float(gn[0]),
            "geometric_mean_den": gd if gd.size > 1 else float(gd[0]),
            "numerator": num, "denominator": den,
            "method": "aitchison_balance",
        },
    )


def cheatsheet():
    return "aitbal: compositions carry only RELATIVE info; balances are scale-invariant; zeros must be replaced first"
