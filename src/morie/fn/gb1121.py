# morie.fn -- function file (rootcoder007/morie)
"""Kendall's tau coefficient."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_kendall_tau"]


def gibbons_kendall_tau(x, y):
    r"""Kendall's tau, the concordance-based association coefficient.

    .. math:: T = \frac{P - Q}{\binom{n}{2}},

    where P and Q count concordant and discordant pairs (Gibbons &
    Chakraborti Ch. 11.2). Counted directly over all pairs, which is
    O(n^2) but exact and transparent; ties contribute to neither P
    nor Q here -- the tie-corrected tau_b lives in
    :mod:`morie.fn.gb1122t`.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Paired observations, n >= 2.

    Returns
    -------
    RichResult
        keys: ``tau``, ``P`` (concordant), ``Q`` (discordant),
        ``n_pairs``, ``n``, ``z`` (null-standardised via the exact
        variance 2(2n+5)/(9n(n-1))), ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.2.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if y.size != n:
        raise ValueError("x and y must have the same length.")
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite.")

    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    prod = dx * dy
    iu = np.triu_indices(n, 1)
    P = int(np.sum(prod[iu] > 0))
    Q = int(np.sum(prod[iu] < 0))
    npairs = n * (n - 1) // 2
    tau = (P - Q) / npairs
    var = 2.0 * (2 * n + 5) / (9.0 * n * (n - 1))
    return RichResult(
        payload={
            "tau": float(tau), "P": P, "Q": Q, "n_pairs": npairs, "n": int(n),
            "z": float(tau / np.sqrt(var)),
            "method": "Kendall tau = (P - Q)/C(n,2) (Gibbons Ch. 11.2)",
        }
    )


def cheatsheet():
    return "gb1121: tau = (P-Q)/C(n,2), exact pair count"
