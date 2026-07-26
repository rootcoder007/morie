# morie.fn -- function file (rootcoder007/morie)
"""Approximate entropy -- Pincus (1991); NOT covered by Rangayyan."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_approximate_entropy"]


def rangayyan_approximate_entropy(x, m=2, r=None):
    """Approximate entropy (Pincus 1991).

    ApEn = φ_m(r) − φ_{m+1}(r), with self-matches INCLUDED (Pincus
    convention) and Chebyshev distance.

    Parameters
    ----------
    x : array-like
    m : int
    r : float, optional
        Tolerance; defaults to 0.2 * std(x).

    Returns
    -------
    RichResult with keys ``ApEn``, ``phi_m``, ``phi_m1``, ``m``, ``r``, ``n``.

    References
    ----------
    Pincus, S. M. (1991). Approximate entropy as a measure of system
        complexity. *Proceedings of the National Academy of Sciences*,
        88(6), 2297-2301. https://doi.org/10.1073/pnas.88.6.2297

    Note: this method is NOT in Rangayyan, contrary to the previous
    docstring's "Ch 7" -- the 2024 edition contains no occurrence of
    "approximate entropy" or "Pincus". The primary paper is the
    specification.

    Unlike sample entropy, ApEn INCLUDES self-matches and evaluates phi^m
    over N-m+1 template vectors and phi^(m+1) over N-m, per Pincus. That
    asymmetry is deliberate here, not the bug that was fixed in rgsam.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if r is None:
        r = 0.2 * x.std(ddof=0)
    r = float(r)
    m = int(m)
    if m + 1 >= N:
        raise ValueError("Need len(x) > m + 1.")

    def _phi(mm: int) -> float:
        nT = N - mm + 1
        templates = np.array([x[i : i + mm] for i in range(nT)])
        d = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        C = (d <= r).sum(axis=1) / nT
        C = np.clip(C, 1e-30, None)
        return float(np.mean(np.log(C)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    apen = float(phi_m - phi_m1)
    res = RichResult(
        title="Approximate entropy",
        summary_lines=[
            ("m", m),
            ("r", r),
            ("N", N),
            ("φ_m", phi_m),
            ("φ_{m+1}", phi_m1),
            ("ApEn", apen),
        ],
        interpretation=f"ApEn = {apen:.4g}. Higher -> more irregular.",
        payload={"ApEn": apen, "phi_m": phi_m, "phi_m1": phi_m1, "m": m, "r": r, "n": N},
    )
    return with_describe_pointer(res, "rgapn")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_approximate_entropy(rng.standard_normal(100), m=2)
# >>> r["ApEn"] > 0
# True


def cheatsheet():
    return "rgapn: approximate entropy -- Pincus 1991 / Richman & Moorman 2000"
