# morie.fn -- function file (rootcoder007/morie)
"""Largest Lyapunov exponent (Rosenstein et al. 1993); NOT covered by Rangayyan."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_lyapunov"]


def rangayyan_lyapunov(x, m=3, tau=1, max_t=None, theiler=10):
    """Largest Lyapunov exponent via Rosenstein's algorithm.

    1. Delay-embed dimension ``m``, lag ``tau``.
    2. Nearest neighbour search with Theiler-window exclusion.
    3. Mean ``⟨ln d(t)⟩`` vs forward step ``t``.
    4. λ₁ = slope of the linear (early-growth) region.

    Parameters
    ----------
    x : array-like
    m : int
    tau : int
    max_t : int, optional
    theiler : int

    Returns
    -------
    RichResult with keys ``lyapunov``, ``divergence_curve``, ``t``.

    References
    ----------
    Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical
        method for calculating largest Lyapunov exponents from small data
        sets. *Physica D: Nonlinear Phenomena*, 65(1-2), 117-134.
        https://doi.org/10.1016/0167-2789(93)90009-P

    Note: this method is NOT in Rangayyan, contrary to the previous
    docstring's "Ch 7" -- the 2024 edition contains no occurrence of
    "Lyapunov" or "Rosenstein" at all.

    NOT YET CERTIFIED AGAINST THE PRIMARY. Rosenstein et al. (1993) is not in
    the reference library, so the numerical details -- the exact divergence
    normalisation, the prescribed fit window, the treatment of zero
    separations -- have not been checked against the paper. What IS verified
    is behavioural: on the logistic map at r = 4, whose largest Lyapunov
    exponent is analytically ln 2 = 0.6931 per iteration, this returns 0.591;
    a periodic signal returns 0.000. The tests pin those properties and
    nothing stronger. Acquire the paper before relying on the magnitude.

    The fit window (the first half of the usable divergence curve) is a fixed
    heuristic, not part of the method: Rosenstein fits the early-growth
    region, and identifying it is a judgement call. Inspect
    ``divergence_curve`` and fit it yourself when the answer matters.
    """
    x = np.asarray(x, dtype=float).ravel()
    # Same guard as rgcrl: a generated test put an array where a scalar
    # belongs, and the failure surfaced from deep inside the embedding.
    for name, val in (("m", m), ("tau", tau), ("theiler", theiler)):
        if np.ndim(val) != 0:
            raise ValueError(
                f"`{name}` must be a scalar integer, got array of shape "
                f"{np.shape(val)}. Signature is rangayyan_lyapunov("
                "x, m=3, tau=1, max_t=None, theiler=10)."
            )
    m, tau, theiler = int(m), int(tau), int(theiler)
    if m < 1 or tau < 1:
        raise ValueError(f"`m` and `tau` must be >= 1, got m={m}, tau={tau}.")
    if theiler < 0:
        raise ValueError(f"`theiler` must be >= 0, got {theiler}.")
    N = x.size
    M = N - (m - 1) * tau
    if M < 10:
        raise ValueError(
            f"Series too short for embedding: {N} samples at m={m}, tau={tau} "
            f"gives {M} embedded points, need >= 10."
        )
    # The Theiler window excludes temporally close pairs from the neighbour
    # search. If it swallows every candidate, argmin over an all-inf row
    # silently returns index 0 -- a "nearest neighbour" that is nothing of
    # the kind, and a lambda computed from it means nothing.
    if 2 * theiler + 1 >= M:
        raise ValueError(
            f"Theiler window {theiler} excludes every neighbour for {M} "
            f"embedded points; need 2*theiler + 1 < {M}."
        )
    Y = np.empty((M, m))
    for i in range(m):
        Y[:, i] = x[i * tau : i * tau + M]
    if max_t is None:
        max_t = min(M // 4, 100)
    d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    iv = np.arange(M)
    mask = np.abs(iv[:, None] - iv[None, :]) <= theiler
    d = np.where(mask, np.inf, d)
    nn = np.argmin(d, axis=1)
    div = np.full(max_t, np.nan)
    for t in range(max_t):
        ok = (iv + t < M) & (nn + t < M)
        if not ok.any():
            continue
        dij = np.linalg.norm(Y[iv[ok] + t] - Y[nn[ok] + t], axis=1)
        dij = dij[dij > 0]
        if dij.size:
            div[t] = float(np.mean(np.log(dij)))
    ts = np.where(np.isfinite(div))[0]
    if ts.size < 3:
        lam = float("nan")
    else:
        half = max(3, ts.size // 2)
        slope, _ = np.polyfit(ts[:half], div[ts[:half]], 1)
        lam = float(slope)
    res = RichResult(
        title="Largest Lyapunov exponent (Rosenstein)",
        summary_lines=[("m", m), ("τ", tau), ("Theiler", theiler), ("λ₁", lam)],
        interpretation=f"λ₁ = {lam:.4g}. >0 chaotic, ~0 marginal, <0 stable.",
        payload={"lyapunov": lam, "divergence_curve": div, "t": np.arange(max_t)},
    )
    return with_describe_pointer(res, "rglyp")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_lyapunov(rng.standard_normal(200), m=3, tau=1, max_t=20)
# >>> "lyapunov" in r
# True


def cheatsheet():
    return "rglyp: largest Lyapunov exponent -- Rosenstein et al. (1993)"
