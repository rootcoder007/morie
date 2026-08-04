# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean-shift: mode-seeking via kernel density gradient ascent."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mean_shift"]

_METHOD = "Mean shift (mode seeking)"


def geron_mean_shift(X, bandwidth, kernel="gaussian", max_iter=300, tol=1e-6, merge_tol=None):
    """
    Mean-shift: mode-seeking via kernel density gradient ascent.

    Formula: x_{t+1} = sum_i K(x_t - x_i) x_i / sum_i K(x_t - x_i)

    Each point climbs the kernel density estimate to a mode; points that
    arrive at the same mode form a cluster.  The number of clusters is
    therefore *not* a parameter -- ``bandwidth`` sets it implicitly, and
    that is the trade: no ``k`` to choose, but a scale that has to be.
    A bandwidth larger than the data spread collapses everything to one
    mode; a tiny one gives one cluster per point.

    The update is a weighted mean, so it never leaves the convex hull of
    the data and the iteration is unconditionally stable -- no learning
    rate appears anywhere despite this being gradient ascent.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    bandwidth : float
        Kernel bandwidth (positive).  Gaussian weights are
        ``exp(-||d||^2 / (2 h^2))``; the flat kernel weights points
        within ``h`` equally.
    kernel : {"gaussian", "flat"}
        Kernel to use.
    max_iter : int
        Iteration cap per starting point.
    tol : float
        Convergence threshold on the shift length.
    merge_tol : float, optional
        Modes closer than this are merged; defaults to ``bandwidth/2``.

    Returns
    -------
    result : RichResult
        Keys: labels, modes, n_clusters, trajectories_iters,
        estimate, n, method.

    Examples
    --------
    Two tight groups 10 apart with a bandwidth of 1 give two modes at
    the group means:

    >>> X = [[0.0], [0.2], [10.0], [10.2]]
    >>> r = geron_mean_shift(X, bandwidth=1.0)
    >>> r["n_clusters"]
    2
    >>> sorted(round(float(v), 4) for v in r["modes"].ravel())
    [0.1, 10.1]

    A bandwidth wider than the data collapses everything into one mode
    at the overall mean:

    >>> w = geron_mean_shift(X, bandwidth=100.0)
    >>> w["n_clusters"], round(float(w["modes"][0, 0]), 4)
    (1, 5.1)

    The flat kernel with a small bandwidth isolates every point:

    >>> f = geron_mean_shift([[0.0], [5.0], [10.0]], bandwidth=1.0, kernel="flat")
    >>> f["n_clusters"]
    3

    References
    ----------
    Géron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_mean_shift: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_mean_shift: X contains non-finite values")
    h = float(bandwidth)
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"geron_mean_shift: bandwidth must be positive and finite, got {bandwidth!r}")
    if kernel not in ("gaussian", "flat"):
        raise ValueError(f"geron_mean_shift: kernel must be 'gaussian' or 'flat', got {kernel!r}")
    if int(max_iter) < 1:
        raise ValueError(f"geron_mean_shift: max_iter must be at least 1, got {max_iter!r}")
    mt = h / 2.0 if merge_tol is None else float(merge_tol)
    if mt <= 0:
        raise ValueError(f"geron_mean_shift: merge_tol must be positive, got {merge_tol!r}")

    m = A.shape[0]
    peaks = np.empty_like(A)
    iters = np.zeros(m, dtype=int)
    for i in range(m):
        x = A[i].copy()
        for step in range(1, int(max_iter) + 1):
            d2 = np.sum((A - x) ** 2, axis=1)
            if kernel == "gaussian":
                w = np.exp(-d2 / (2.0 * h * h))
            else:
                w = (d2 <= h * h).astype(float)
            tot = float(np.sum(w))
            if tot == 0:
                raise ValueError(
                    f"geron_mean_shift: point {i} has no neighbours within the flat bandwidth {h}; "
                    f"increase bandwidth or use the gaussian kernel"
                )
            new = (w @ A) / tot
            shift = float(np.linalg.norm(new - x))
            x = new
            iters[i] = step
            if shift <= float(tol):
                break
        peaks[i] = x

    # Merge modes that landed within merge_tol of one another.
    modes = []
    labels = np.full(m, -1, dtype=int)
    for i in range(m):
        for j, mode in enumerate(modes):
            if np.linalg.norm(peaks[i] - mode) <= mt:
                labels[i] = j
                break
        else:
            labels[i] = len(modes)
            modes.append(peaks[i])
    modes = np.asarray(modes)

    # Refine each mode to the mean of its members.
    for j in range(modes.shape[0]):
        modes[j] = A[labels == j].mean(axis=0) if np.any(labels == j) else modes[j]

    return RichResult(
        title="Mean shift",
        summary_lines=[("Bandwidth", h), ("Kernel", kernel), ("Modes found", int(modes.shape[0]))],
        interpretation=(
            "The cluster count is set by the bandwidth, not chosen: widen it and modes merge, "
            "narrow it and every point becomes its own mode."
        ),
        payload={
            "labels": labels,
            "modes": modes,
            "n_clusters": int(modes.shape[0]),
            "peaks": peaks,
            "trajectories_iters": iters,
            "bandwidth": h,
            "estimate": float(modes.shape[0]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmnsh: mean shift -- kernel-weighted mean updates to a density mode; bandwidth sets k"


# compact alias per ledger/NAMING.md
geronmeanshift = geron_mean_shift
