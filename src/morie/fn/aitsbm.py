# morie.fn -- function file (rootcoder007/morie)
"""Spurious correlation diagnostic via subcompositional incoherence."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["compositional_simbias"]


def _close(X):
    """Close each row to unit sum -- the closure operator of the simplex."""
    return X / X.sum(axis=1, keepdims=True)


def compositional_simbias(X, idx):
    r"""Spurious correlation of compositional parts, and its cure.

    Pearson (1897) showed that correlations between ratios sharing a
    denominator are constrained by the denominator, not by any relation
    among the numerators. Compositional data carry exactly that defect:
    the parts are closed to a constant sum, so their correlations are
    driven by the closure. The visible symptom is *subcompositional
    incoherence* -- the correlation between two parts changes when other,
    unrelated parts are dropped from the composition.

    This diagnostic measures that change directly. It takes the same two
    parts, computes their Pearson correlation in the full composition and
    again after re-closing to the subcomposition named by ``idx``, and
    reports the difference. A large ``delta`` means a correlation read off
    the raw parts says more about which parts happened to be measured
    than about the parts themselves.

    Alongside it the function returns the variation-array entry,

    .. math::

        \tau_{ij} = \mathrm{var}\!\left(\log \frac{x_i}{x_j}\right)

    which is invariant under closure -- the log-ratio cancels any common
    factor -- so it takes the *same* value in the full composition and in
    every subcomposition containing both parts. ``tau_delta`` is
    therefore zero up to floating-point error, and that is the point: it
    is the coherent quantity to report where the raw correlation is not.

    Parameters
    ----------
    X : array-like, shape (n, D)
        Compositional data, D >= 3 parts. All entries must be strictly
        positive; the rows need not already sum to a constant, since they
        are closed on entry.
    idx : array-like of int
        Indices of the parts forming the subcomposition, length >= 2 and
        < D. The first two entries are the pair whose correlation is
        compared -- with the whole subcomposition equal to the whole
        composition there would be nothing to compare.

    Returns
    -------
    RichResult
        keys: ``rho_full``, ``rho_sub``, ``delta`` (``rho_sub -
        rho_full``), ``tau_full``, ``tau_sub``, ``tau_delta``, ``pair``,
        ``idx``, ``n``, ``D``, ``method``.

    References
    ----------
    Aitchison, J. (1986). *The Statistical Analysis of Compositional
    Data*. Monographs on Statistics and Applied Probability. Chapman &
    Hall, London, 416 pp.

    Pearson, K. (1897). Mathematical contributions to the theory of
    evolution -- on a form of spurious correlation which may arise when
    indices are used in the measurement of organs. *Proceedings of the
    Royal Society of London*, 60, 489-498.
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError("X must be two-dimensional (n observations x D parts)")
    n, D = Xa.shape
    if D < 3:
        raise ValueError(f"Need at least 3 parts to form a proper subcomposition, got D={D}.")
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")
    if not np.all(Xa > 0):
        raise ValueError("Compositional parts must be strictly positive; log-ratios are undefined at zero.")

    sub = np.asarray(idx, dtype=int).ravel()
    if sub.size < 2:
        raise ValueError(f"idx must name at least 2 parts, got {sub.size}.")
    if sub.size >= D:
        raise ValueError(f"idx must name fewer than D={D} parts, else the subcomposition is the composition.")
    if np.any(sub < 0) or np.any(sub >= D):
        raise ValueError(f"idx entries must lie in [0, {D - 1}].")
    if np.unique(sub).size != sub.size:
        raise ValueError("idx must not repeat a part.")

    i, j = int(sub[0]), int(sub[1])
    full = _close(Xa)
    subc = _close(Xa[:, sub])

    rho_full = float(np.corrcoef(full[:, i], full[:, j])[0, 1])
    rho_sub = float(np.corrcoef(subc[:, 0], subc[:, 1])[0, 1])

    tau_full = float(np.var(np.log(full[:, i] / full[:, j])))
    tau_sub = float(np.var(np.log(subc[:, 0] / subc[:, 1])))

    return RichResult(
        title="Subcompositional incoherence of the raw correlation",
        payload={
            "rho_full": rho_full,
            "rho_sub": rho_sub,
            "delta": rho_sub - rho_full,
            "statistic": rho_sub - rho_full,
            "tau_full": tau_full,
            "tau_sub": tau_sub,
            "tau_delta": tau_sub - tau_full,
            "pair": (i, j),
            "idx": sub,
            "n": int(n),
            "D": int(D),
            "method": "Pearson correlation under closure vs. Aitchison variation array",
        },
    )


def cheatsheet():
    return "aitsbm: Spurious correlation diagnostic via subcompositional incoherence"
