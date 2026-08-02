# morie.fn -- function file (rootcoder007/morie)
"""Sample entropy -- Richman & Moorman (2000); NOT covered by Rangayyan."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_sample_entropy"]


def rangayyan_sample_entropy(x, m=2, r=None):
    """Sample entropy (Richman & Moorman 2000).

    SampEn = −ln(A / B) where ``B`` is the number of template-vector pairs
    of length ``m`` with Chebyshev distance ≤ ``r`` and ``A`` is the
    count at length ``m+1`` (self-matches excluded, i<j).

    Parameters
    ----------
    x : array-like
    m : int
    r : float, optional
        Tolerance; defaults to 0.2 * std(x).

    Returns
    -------
    RichResult with keys ``SampEn``, ``A``, ``B``, ``m``, ``r``, ``n``.

    References
    ----------
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
        analysis using approximate entropy and sample entropy. *American
        Journal of Physiology - Heart and Circulatory Physiology*, 278(6),
        H2039-H2049. https://doi.org/10.1152/ajpheart.2000.278.6.H2039
    PhysioNet, "sampen: sample entropy estimation".
        https://physionet.org/physiotools/sampen/

    Note: this method is NOT in Rangayyan, contrary to the previous
    docstring's "Ch 7" -- the 2024 edition contains no occurrence of
    "sample entropy", "approximate entropy", "Pincus" or "Richman". The
    primary paper is the specification.

    SampEn differs from ApEn in exactly two ways, and both are load-bearing
    here: self-matches are EXCLUDED (i < j), and the length-m and
    length-(m+1) counts are taken over the SAME N-m template vectors so that
    A and B share a denominator.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if r is None:
        r = 0.2 * x.std(ddof=0)
    r = float(r)
    m = int(m)
    if m + 1 >= N:
        raise ValueError("Need len(x) > m + 1.")

    # Richman & Moorman count BOTH the length-m and the length-(m+1) matches
    # over the SAME set of N-m template vectors. Using N-mm+1 per call gave B
    # one extra template that A could not have, so the two counts had different
    # denominators -- reintroducing precisely the bias SampEn was defined to
    # remove. Measured on 300 Gaussian samples (m=2, r=0.2 sd): B picked up 9
    # spurious pairs and SampEn was biased upward by +0.018.
    n_templates = N - m
    if n_templates < 2:
        raise ValueError("Need len(x) > m + 1.")

    def _matches(mm: int) -> int:
        templates = np.array([x[i : i + mm] for i in range(n_templates)])
        d = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        # Self-matches excluded (i < j) -- the second way SampEn differs from
        # ApEn, which includes them.
        mask = np.triu(np.ones_like(d, dtype=bool), k=1)
        return int(np.sum((d <= r) & mask))

    B = _matches(m)
    A = _matches(m + 1)
    sampen = float("inf") if (A == 0 or B == 0) else float(-np.log(A / B))
    res = RichResult(
        title="Sample entropy",
        summary_lines=[
            ("m", m),
            ("r", r),
            ("N", N),
            ("B (matches m)", B),
            ("A (matches m+1)", A),
            ("SampEn", sampen),
        ],
        interpretation=f"SampEn = {sampen:.4g}. Higher -> more irregular.",
        payload={"SampEn": sampen, "A": A, "B": B, "m": m, "r": r, "n": N},
    )
    return with_describe_pointer(res, "rgsam")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_sample_entropy(rng.standard_normal(100), m=2)
# >>> r["SampEn"] > 0
# True


def cheatsheet():
    return "rgsam: sample entropy -- Pincus 1991 / Richman & Moorman 2000"
