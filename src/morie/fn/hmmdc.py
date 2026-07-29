# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mode collapse: GAN generator outputs limited variety."""

import numpy as np

from ._richresult import RichResult
from .hmmds import pairwise_distances

__all__ = ["geron_mode_collapse"]

_METHOD = "Mode-collapse diagnostics"


def geron_mode_collapse(samples, reference=None, tol=None):
    """
    Mode collapse: GAN generator outputs limited variety.

    Formula: symptom: low sample diversity relative to data

    Mode collapse is not visible in the loss -- a generator producing one
    perfect digit can keep a discriminator honest indefinitely -- so it
    has to be measured on the samples.  Three numbers, each of which
    catches a different flavour:

    ``n_modes`` -- distinct sample clusters at tolerance ``tol``
    (single-linkage on the pairwise distances).  Total collapse is 1.

    ``mean_pairwise_distance`` -- average distance between generated
    samples.  Falls toward 0 as the generator narrows.

    ``coverage`` -- with a ``reference`` set of real data, the fraction
    of *real* modes that have a generated sample nearest to them.  This
    is the one that matters: a generator can produce several distinct
    outputs and still miss most of the data's modes, and the first two
    numbers would call that healthy.

    ``collapse_score`` is ``1 - n_modes/n_samples``, so 1 is total
    collapse and 0 is all-distinct.

    ``tol`` defaults to 5% of the largest pairwise distance in the
    samples; with a reference set the reference's spread is used
    instead, which is the more honest scale.

    Parameters
    ----------
    samples : array-like, shape (m, n)
        Generated samples.
    reference : array-like, shape (r, n), optional
        Real data, for the coverage measure.
    tol : float, optional
        Distance below which two samples count as the same mode.

    Returns
    -------
    result : RichResult
        Keys: n_modes, collapse_score, mean_pairwise_distance,
        coverage, mode_labels, mode_sizes, estimate, n, method.

    Examples
    --------
    Identical samples: one mode, total collapse, zero spread.

    >>> r = geron_mode_collapse([[1.0], [1.0], [1.0], [1.0]])
    >>> r["n_modes"], float(r["collapse_score"]), float(r["mean_pairwise_distance"])
    (1, 0.75, 0.0)

    Four well-separated samples: four modes, no collapse.

    >>> d = geron_mode_collapse([[0.0], [10.0], [20.0], [30.0]])
    >>> d["n_modes"], float(d["collapse_score"])
    (4, 0.0)

    Coverage catches the case the other measures miss: two distinct
    generated values against four real modes cover only half of them.

    >>> real = [[0.0], [10.0], [20.0], [30.0]]
    >>> c = geron_mode_collapse([[0.1], [10.1], [0.2], [9.9]], reference=real)
    >>> float(c["coverage"])
    0.5
    >>> c["n_modes"]
    2

    Full coverage when every real mode is hit:

    >>> f = geron_mode_collapse([[0.1], [10.1], [19.9], [30.2]], reference=real)
    >>> float(f["coverage"])
    1.0

    References
    ----------
    Géron Ch 18
    """
    S = np.asarray(samples, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.ndim != 2 or S.size == 0:
        raise ValueError(f"geron_mode_collapse: samples must be a non-empty 2-D array, got shape {S.shape}")
    if S.shape[0] < 2:
        raise ValueError("geron_mode_collapse: diversity needs at least 2 samples")
    if not np.all(np.isfinite(S)):
        raise ValueError("geron_mode_collapse: samples contain non-finite values")

    D = pairwise_distances(S)
    m = S.shape[0]
    iu = np.triu_indices(m, 1)
    mean_pd = float(np.mean(D[iu]))
    max_pd = float(np.max(D[iu]))

    R = None
    if reference is not None:
        R = np.asarray(reference, dtype=float)
        if R.ndim == 1:
            R = R.reshape(-1, 1)
        if R.ndim != 2 or R.size == 0:
            raise ValueError(f"geron_mode_collapse: reference must be a non-empty 2-D array, got shape {R.shape}")
        if R.shape[1] != S.shape[1]:
            raise ValueError(
                f"geron_mode_collapse: reference has {R.shape[1]} features but samples have {S.shape[1]}"
            )
        if not np.all(np.isfinite(R)):
            raise ValueError("geron_mode_collapse: reference contains non-finite values")

    if tol is None:
        if R is not None and R.shape[0] > 1:
            Dr = pairwise_distances(R)
            scale = float(np.max(Dr[np.triu_indices(R.shape[0], 1)]))
        else:
            scale = max_pd
        t = 0.05 * scale
    else:
        t = float(tol)
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"geron_mode_collapse: tol must be finite and non-negative, got {tol!r}")

    # Single-linkage grouping at radius t.
    labels = np.full(m, -1, dtype=int)
    n_modes = 0
    for i in range(m):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = n_modes
        while stack:
            u = stack.pop()
            for v in np.flatnonzero((D[u] <= t) & (labels == -1)):
                labels[int(v)] = n_modes
                stack.append(int(v))
        n_modes += 1
    sizes = np.bincount(labels, minlength=n_modes)

    coverage = None
    if R is not None:
        # A real point is covered if some generated sample is within t of it.
        Dg = np.sqrt(np.clip(np.sum((R[:, None, :] - S[None, :, :]) ** 2, axis=2), 0.0, None))
        covered = np.any(Dg <= max(t, 1e-12) * 5.0, axis=1) if t > 0 else np.any(Dg == 0, axis=1)
        coverage = float(np.mean(covered))

    collapse = 1.0 - n_modes / m

    warns = []
    if n_modes == 1:
        warns.append("every sample falls in a single mode: the generator has collapsed completely.")
    if coverage is not None and coverage < 0.5:
        warns.append(
            f"only {coverage:.0%} of the reference modes are covered; the samples may look diverse "
            f"and still miss most of the data distribution."
        )

    return RichResult(
        title="Mode collapse",
        summary_lines=[
            ("Samples", int(m)),
            ("Distinct modes", int(n_modes)),
            ("Collapse score", collapse),
            ("Mean pairwise distance", mean_pd),
            ("Coverage", coverage if coverage is not None else "n/a"),
        ],
        warnings=warns,
        interpretation=(
            "The loss cannot see mode collapse; coverage against real data is the measure that "
            "catches a generator that is varied but still missing most modes."
        ),
        payload={
            "n_modes": int(n_modes),
            "collapse_score": collapse,
            "mean_pairwise_distance": mean_pd,
            "max_pairwise_distance": max_pd,
            "coverage": coverage,
            "mode_labels": labels,
            "mode_sizes": sizes,
            "tol": t,
            "estimate": collapse,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmdc: mode-collapse diagnostics -- distinct modes, pairwise spread, and coverage of real modes"
