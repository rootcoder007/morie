# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mode-collapse metric for generative samples."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_gan_mode_collapse_metric"]

_METHOD = "Mode coverage / collapse rate"


def geron_gan_mode_collapse_metric(samples, true_modes, tol=None):
    r"""What fraction of the real distribution's modes the generator found.

    .. math::
        \text{coverage} = \frac{|\text{modes hit}|}{|\text{modes true}|},
        \qquad \text{collapse rate} = 1 - \text{coverage}

    A sample counts as hitting the mode it is nearest to, provided it is
    within ``tol``; samples further than that from every mode are
    off-distribution and counted separately rather than being credited
    to the closest one.

    Coverage alone cannot be trusted, which is why
    ``samples_per_mode`` is returned: a generator that emits 999 copies
    of one mode and one of another scores full coverage on two modes
    while being obviously collapsed.  The counts show it immediately.

    Parameters
    ----------
    samples : array-like, shape (m, d) or (m,)
        Generated samples.
    true_modes : array-like, shape (K, d) or (K,)
        Known mode locations.
    tol : float, optional
        Radius counting as a hit. Defaults to half the smallest
        distance between two true modes, so the balls cannot overlap.

    Returns
    -------
    RichResult
        Payload keys ``coverage``, ``mode_collapse_rate``,
        ``modes_hit``, ``modes_missed``, ``samples_per_mode``,
        ``n_off_distribution``, ``tol``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 18, Training Difficulties / Mode Collapse section.

    Examples
    --------
    Two real modes, every sample sitting on the first: half the modes
    are covered, so the collapse rate is 0.5.

    >>> r = geron_gan_mode_collapse_metric([[0.1], [0.2], [-0.1]], [[0.0], [10.0]])
    >>> r["coverage"], r["mode_collapse_rate"]
    (0.5, 0.5)
    >>> r["modes_missed"]
    [1]
    >>> r["samples_per_mode"]
    [3, 0]

    Reach both modes and the collapse rate goes to zero -- but the
    counts still show how lopsided the generator is:

    >>> r2 = geron_gan_mode_collapse_metric([[0.0], [0.0], [10.0]], [[0.0], [10.0]])
    >>> r2["coverage"], r2["samples_per_mode"]
    (1.0, [2, 1])

    A sample far from everything is not credited to the nearest mode:

    >>> r3 = geron_gan_mode_collapse_metric([[5.0]], [[0.0], [10.0]], tol=1.0)
    >>> r3["n_off_distribution"], r3["coverage"]
    (1, 0.0)
    """
    S = np.atleast_2d(np.asarray(samples, dtype=float))
    M = np.atleast_2d(np.asarray(true_modes, dtype=float))
    if S.shape[0] == 1 and M.shape[1] != S.shape[1] and S.shape[1] == M.shape[0]:
        S = S.T
    if S.shape[1] != M.shape[1]:
        raise ValueError(
            f"samples have {S.shape[1]} dimensions but modes have {M.shape[1]}."
        )
    if S.size == 0 or M.size == 0:
        raise ValueError("samples and true_modes must both be non-empty.")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(M)):
        raise ValueError("samples and true_modes must be finite.")
    K = M.shape[0]

    if tol is None:
        if K < 2:
            raise ValueError(
                "with a single true mode there is no inter-mode distance to derive "
                "tol from; pass tol explicitly."
            )
        i, j = np.triu_indices(K, k=1)
        sep = np.linalg.norm(M[i] - M[j], axis=1)
        if np.any(sep == 0):
            raise ValueError("true_modes contains duplicates; modes must be distinct.")
        tol = float(sep.min() / 2.0)
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be a positive finite radius, got {tol}.")

    D = np.linalg.norm(S[:, None, :] - M[None, :, :], axis=2)
    nearest = D.argmin(axis=1)
    within = D.min(axis=1) <= tol
    counts = np.array([int(np.sum(within & (nearest == k))) for k in range(K)])
    hit = np.flatnonzero(counts > 0)
    coverage = float(hit.size) / float(K)

    return RichResult(
        title="Mode coverage / collapse",
        summary_lines=[("Coverage", coverage), ("Collapse rate", 1.0 - coverage),
                       ("Modes", int(K))],
        payload={
            "coverage": coverage,
            "mode_collapse_rate": 1.0 - coverage,
            "modes_hit": hit.tolist(),
            "modes_missed": np.flatnonzero(counts == 0).tolist(),
            "samples_per_mode": counts.tolist(),
            "n_off_distribution": int(np.sum(~within)),
            "tol": tol,
            "estimate": coverage,
            "n": int(S.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmcol: coverage = modes hit / modes true; per-mode counts expose lopsided 'full' coverage"
