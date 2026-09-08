# morie.fn -- function file (rootcoder007/morie)
"""Correlation dimension (Grassberger-Procaccia 1983); NOT covered by Rangayyan."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_correlation_dimension"]


def rangayyan_correlation_dimension(x, m=3, tau=1, n_r=20):
    """Correlation dimension D₂ (Grassberger-Procaccia 1983).

    1. Delay-embed ``x`` to dimension ``m`` with lag ``tau``.
    2. Correlation sum::

           C(r) = (1/(M(M-1))) Σ_{i≠j} Θ(r - ||Y_i - Y_j||)

    3. D₂ = slope of log C(r) vs log r in the scaling region.

    Parameters
    ----------
    x : array-like
    m : int
        Embedding dimension.
    tau : int
        Embedding lag.
    n_r : int
        Number of radii.

    Returns
    -------
    RichResult with keys ``D2``, ``log_r``, ``log_C``, ``m``, ``tau``.

    References
    ----------
    Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of
        strange attractors. *Physica D: Nonlinear Phenomena*, 9(1-2),
        189-208. https://doi.org/10.1016/0167-2789(83)90298-1
        (PRIMARY -- now in the library.)

    Note: this method is NOT in Rangayyan, contrary to the previous
    docstring's "Ch 7". The 2024 edition mentions "correlation dimension"
    exactly once, as a citation inside a sentence, and contains no occurrence
    of "Grassberger", "Procaccia" or "correlation sum". The primary paper is
    the specification.

    The correlation sum is the Grassberger-Procaccia estimator

        C_hat(r) = 2 / (M(M-1)) * sum_{i<j} theta(r - ||Y_i - Y_j||)

    with theta the Heaviside step, theta(u) = 1 for u >= 0. Averaging the
    indicator over the M(M-1)/2 unordered pairs gives exactly this
    normalisation.

    Note on the scaling region: D2 is the slope of log C(r) against log r
    *in the scaling region*, and identifying that region is a judgement call
    the paper leaves to the analyst. This implementation uses a fixed
    heuristic -- discard the lowest and highest fifth of the usable radii --
    which is a convenience, not part of the method. Inspect ``log_r`` and
    ``log_C`` and fit the region yourself when the answer matters.
    """
    x = np.asarray(x, dtype=float).ravel()
    # m, tau and n_r are scalars. Passing an array (a generated test called
    # this as f(x, y), putting a whole series where the embedding dimension
    # goes) otherwise surfaced as "truth value of an array ... is ambiguous"
    # from deep inside the embedding, which says nothing about the real
    # mistake.
    for name, val in (("m", m), ("tau", tau), ("n_r", n_r)):
        if np.ndim(val) != 0:
            raise ValueError(
                f"`{name}` must be a scalar integer, got array of shape "
                f"{np.shape(val)}. Signature is "
                "rangayyan_correlation_dimension(x, m=3, tau=1, n_r=20)."
            )
    m, tau, n_r = int(m), int(tau), int(n_r)
    if m < 1 or tau < 1:
        raise ValueError(f"`m` and `tau` must be >= 1, got m={m}, tau={tau}.")
    N = x.size
    M = N - (m - 1) * tau
    if M < 10:
        raise ValueError(
            f"Series too short for embedding: {N} samples at m={m}, tau={tau} "
            f"gives {M} embedded points, need >= 10."
        )
    Y = np.empty((M, m))
    for i in range(m):
        Y[:, i] = x[i * tau : i * tau + M]
    d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    iu = np.triu_indices(M, k=1)
    dist = d[iu]
    if dist.size == 0:
        raise ValueError("No pairwise distances.")
    pos = dist[dist > 0]
    rmin = max(pos.min() if pos.size else 1e-12, 1e-12)
    rmax = dist.max()
    rs = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([np.mean(dist <= r) for r in rs])
    # Require a minimum pair count per radius, not merely C > 0.
    #
    # rmin is the smallest pairwise distance, so C at the low end is estimated
    # from one or two pairs -- 1/6903 for a 118-point embedding. That is noise,
    # not an estimate, and it made D2 depend on the *scale* of the input: an
    # affine rescale perturbs the boundary pair in or out, changing how many
    # radii pass the mask, which shifts the fit window and moves the slope.
    # Measured: D2 2.856 -> 2.753 under x -> 100x, on a quantity that is a
    # dimension and must be invariant.
    #
    # Ten pairs is a floor, not a derived threshold; the paper leaves scaling-
    # region choice to the analyst. It is enough to make the usable set stable
    # under rescaling, which is the property being restored.
    n_pairs = C * dist.size
    mask = (n_pairs >= 10) & np.isfinite(C)
    log_r = np.log(rs[mask])
    log_C = np.log(C[mask])
    if log_r.size < 3:
        D2 = float("nan")
    else:
        n = log_r.size
        lo = max(1, n // 5)
        hi = max(lo + 2, n - n // 5)
        slope, _ = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)
        D2 = float(slope)
    res = RichResult(
        title="Correlation dimension (Grassberger-Procaccia)",
        summary_lines=[("m", m), ("τ", tau), ("D₂", D2)],
        interpretation=f"D₂ = {D2:.4g}. Saturates with m for low-dim chaos.",
        payload={"D2": D2, "log_r": log_r, "log_C": log_C, "m": m, "tau": tau},
    )
    return with_describe_pointer(res, "rgcrl")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_correlation_dimension(rng.standard_normal(200), m=3, tau=1, n_r=15)
# >>> np.isfinite(r["D2"])
# True


def cheatsheet():
    return "rgcrl: correlation dimension D₂ -- Grassberger & Procaccia (1983)"
