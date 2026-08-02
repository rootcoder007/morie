# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learning-rate heuristic: start with LR finder, use 1/10 of divergence point."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_learning_rate_heuristic"]

_METHOD = "LR-finder heuristic (divergence point / 10)"


def geron_learning_rate_heuristic(lr_curve, divergence_factor=4.0, safety=10.0):
    """
    Learning-rate heuristic: start with LR finder, use 1/10 of divergence point.

    Formula: lr = lr_diverge / 10

    Given an LR-finder sweep -- learning rate raised geometrically over a
    few hundred batches, loss recorded at each -- the divergence point is
    the first rate whose loss exceeds ``divergence_factor`` times the
    best loss seen so far, and the recommendation is that rate divided by
    ``safety`` (10 by default).

    The rate at the *minimum* of the curve is deliberately not the
    recommendation: by the time the loss has bottomed out the rate is
    already close to unstable, and training a whole run there diverges.
    Both are returned so the gap is visible.

    ``lr_curve`` is ``(lr, loss)`` pairs, or a mapping, or two parallel
    sequences.  Rates must be positive and strictly increasing -- an
    unsorted sweep would make "the first rate that diverges" meaningless.

    Parameters
    ----------
    lr_curve : array-like, shape (n, 2), or mapping, or (lrs, losses)
        The sweep.
    divergence_factor : float
        Multiple of the running-minimum loss that counts as divergence
        (> 1).
    safety : float
        Divisor applied to the divergence rate (>= 1).

    Returns
    -------
    result : RichResult
        Keys: lr, lr_diverge, lr_min_loss, min_loss, diverged,
        estimate, n, method.

    Examples
    --------
    A loss that falls then blows up: the minimum is at lr = 0.01 and the
    first rate exceeding 4x the running minimum is 0.1, so the
    recommendation is 0.01.

    >>> curve = [(1e-4, 2.0), (1e-3, 1.0), (1e-2, 0.5), (1e-1, 4.0), (1.0, 50.0)]
    >>> r = geron_learning_rate_heuristic(curve)
    >>> float(r["lr_diverge"]), round(float(r["lr"]), 12)
    (0.1, 0.01)
    >>> float(r["lr_min_loss"]), float(r["min_loss"])
    (0.01, 0.5)

    Note that the recommendation is a factor of 10 *below* the minimum's
    rate here only by coincidence of the grid; what it is always below is
    the divergence point:

    >>> bool(r["lr"] < r["lr_diverge"])
    True

    A sweep that never diverges falls back to the rate at the minimum
    and says so:

    >>> flat = geron_learning_rate_heuristic([(1e-4, 2.0), (1e-3, 1.5), (1e-2, 1.0)])
    >>> flat["diverged"], float(flat["lr"])
    (False, 0.01)

    A non-increasing sweep is refused:

    >>> geron_learning_rate_heuristic([(1e-2, 1.0), (1e-3, 2.0)])
    Traceback (most recent call last):
        ...
    ValueError: geron_learning_rate_heuristic: learning rates must be strictly increasing; entry 1 (0.001) is not above entry 0 (0.01)

    References
    ----------
    Géron Ch 9
    """
    if hasattr(lr_curve, "items"):
        pairs = sorted(((float(k), float(v)) for k, v in lr_curve.items()))
        lrs = np.asarray([p[0] for p in pairs])
        losses = np.asarray([p[1] for p in pairs])
    else:
        arr = np.asarray(lr_curve, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            lrs, losses = arr[:, 0], arr[:, 1]
        elif arr.ndim == 2 and arr.shape[0] == 2:
            lrs, losses = arr[0], arr[1]
        else:
            raise ValueError(
                f"geron_learning_rate_heuristic: lr_curve must be (lr, loss) pairs or a mapping, got shape {arr.shape}"
            )
    if lrs.size < 2:
        raise ValueError(f"geron_learning_rate_heuristic: the sweep needs at least 2 points, got {lrs.size}")
    if not np.all(np.isfinite(lrs)) or not np.all(np.isfinite(losses)):
        raise ValueError("geron_learning_rate_heuristic: lr_curve contains non-finite values")
    if np.any(lrs <= 0):
        raise ValueError("geron_learning_rate_heuristic: learning rates must be positive")
    bad = np.flatnonzero(np.diff(lrs) <= 0)
    if bad.size:
        i = int(bad[0]) + 1
        raise ValueError(
            f"geron_learning_rate_heuristic: learning rates must be strictly increasing; "
            f"entry {i} ({lrs[i]:g}) is not above entry {i - 1} ({lrs[i - 1]:g})"
        )
    df = float(divergence_factor)
    if not np.isfinite(df) or df <= 1:
        raise ValueError(f"geron_learning_rate_heuristic: divergence_factor must exceed 1, got {divergence_factor!r}")
    sf = float(safety)
    if not np.isfinite(sf) or sf < 1:
        raise ValueError(f"geron_learning_rate_heuristic: safety must be at least 1, got {safety!r}")

    running_min = np.minimum.accumulate(losses)
    i_min = int(np.argmin(losses))
    diverge_i = None
    for i in range(1, losses.size):
        if losses[i] > df * running_min[i - 1]:
            diverge_i = i
            break

    if diverge_i is None:
        lr_div = float("nan")
        lr = float(lrs[i_min])
        diverged = False
    else:
        lr_div = float(lrs[diverge_i])
        lr = lr_div / sf
        diverged = True

    return RichResult(
        title="Learning-rate finder",
        summary_lines=[
            ("Recommended lr", lr),
            ("Divergence lr", lr_div),
            ("lr at minimum loss", float(lrs[i_min])),
            ("Minimum loss", float(losses[i_min])),
        ],
        warnings=(
            []
            if diverged
            else [
                "the sweep never diverged, so the divergence point is unknown; the rate at the "
                "minimum loss was used instead and may still be too high. Extend the sweep upward."
            ]
        ),
        interpretation=(
            "Do not train at the rate that minimises the finder curve -- by then the run is already "
            "close to unstable; back off by an order of magnitude from divergence."
        ),
        payload={
            "lr": lr,
            "lr_diverge": lr_div,
            "lr_min_loss": float(lrs[i_min]),
            "min_loss": float(losses[i_min]),
            "diverged": diverged,
            "lrs": lrs,
            "losses": losses,
            "estimate": lr,
            "n": int(lrs.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlrh: LR finder -- first rate exceeding 4x the running-min loss, divided by 10"
