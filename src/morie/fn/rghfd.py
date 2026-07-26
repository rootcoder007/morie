# morie.fn -- function file (rootcoder007/morie)
"""Higuchi fractal dimension -- Rangayyan Sec. 5.13.2, eqs (5.39)-(5.41)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_higuchi_fd"]


def rangayyan_higuchi_fd(x, kmax=10):
    """Higuchi (1988) fractal dimension.

    Eq (5.39)  x_k(m) = x(m), x(m+k), x(m+2k), ..., x(m + floor((N-m)/k) k),
               for m = 1, 2, ..., k  (1-based).
    Eq (5.40)  L(m,k) = (1/k) * (N-1) / (k floor((N-m)/k))
                        * sum_{i=1}^{floor((N-m)/k)} |x(m+ik) - x(m+(i-1)k)|
    Eq (5.41)  L(k)   = (1/k) sum_{m=1}^{k} L(m,k)

    FD is the slope of a straight-line fit to a log-log plot of L(k)
    against 1/k.

    Parameters
    ----------
    x : array-like
    kmax : int
        Maximum time lag.

    Returns
    -------
    RichResult with keys ``HFD``, ``log_L``, ``log_inv_k``, ``kmax``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. *Biomedical Signal Analysis*,
        3rd ed. (IEEE Press / Wiley, 2024),
        Sec. 5.13.2 "Higuchi's method", p. 304, eqs (5.39)-(5.41).
    Higuchi, T. (1988). Approach to an irregular time series on the basis of
        the fractal theory. *Physica D*, 31, 277-283.

    Note: the docstring previously cited Ch. 7; Higuchi's method is in
    Sec. 5.13.2 of the 2024 edition, verified against the typeset PDF.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 4 or kmax < 2:
        raise ValueError("Need len(x) >= 4 and kmax >= 2.")
    kmax = int(min(kmax, N // 2))
    L = np.empty(kmax)
    for k in range(1, kmax + 1):
        lk = []
        for m in range(1, k + 1):
            # Eq (5.39): x_k(m) starts at the m-th sample, 1-based in the book,
            # so index m-1 into the 0-based array.
            idx = np.arange(m - 1, N, k)
            if idx.size < 2:
                continue
            diffs = np.sum(np.abs(np.diff(x[idx])))
            # Eq (5.40): the normaliser is floor((N - m)/k) with the book's
            # 1-based m, and it must equal the number of difference terms
            # actually summed. The previous code passed the 0-based loop index
            # here, making the denominator floor((N - m + 1)/k) while the
            # numerator still had floor((N - m)/k) terms -- the two disagreed
            # whenever (N - m) was not a multiple of k. Deriving it from
            # idx.size keeps them identical by construction.
            n_terms = idx.size - 1
            norm = (N - 1) / (k * n_terms)
            lk.append((diffs / k) * norm)
        L[k - 1] = np.mean(lk) if lk else np.nan
    ks = np.arange(1, kmax + 1)
    log_L = np.log(L)
    log_inv_k = np.log(1.0 / ks)
    slope, intercept = np.polyfit(log_inv_k, log_L, 1)
    res = RichResult(
        title="Higuchi fractal dimension",
        summary_lines=[("HFD", float(slope)), ("kmax", kmax), ("N", N)],
        interpretation=f"HFD = {slope:.4g}. ~1 smooth, ~2 rough.",
        payload={
            "HFD": float(slope),
            "intercept": float(intercept),
            "log_L": log_L,
            "log_inv_k": log_inv_k,
            "kmax": kmax,
        },
    )
    return with_describe_pointer(res, "rghfd")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_higuchi_fd(rng.standard_normal(500), kmax=8)
# >>> 1.0 <= r["HFD"] <= 2.5
# True


def cheatsheet():
    return "rghfd: Higuchi fractal dimension -- Rangayyan Sec. 5.13.2"
