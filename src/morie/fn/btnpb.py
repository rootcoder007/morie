# morie.fn -- function file (rootcoder007/morie)
"""Non-overlapping block bootstrap."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boot_nonoverlap_block"]


def boot_nonoverlap_block(x, block_len=None, stat=None, B=500, seed=0):
    r"""Bootstrap a dependent series by resampling contiguous blocks.

    The ordinary bootstrap resamples observations independently, which
    destroys the very dependence a time series is defined by and yields
    standard errors that are far too small. The block bootstrap resamples
    contiguous blocks instead, preserving dependence **within** a block while
    treating blocks as exchangeable.

    Block length is the whole design decision. Too short and dependence is
    destroyed again; too long and there are too few blocks to resample. The
    default :math:`\lfloor n^{1/3} \rfloor` is the standard rate for the
    non-overlapping case.

    Non-overlapping blocks partition the series, so blocks are genuinely
    independent under the model but the effective number of resampling units
    is only :math:`n/\ell`. The moving-block variant reuses overlapping
    windows and is more efficient at the cost of correlated blocks; this is
    the simpler, more conservative choice, and it is worth knowing which one
    is in use when comparing interval widths.

    Parameters
    ----------
    x : array-like
        Series.
    block_len : int, optional
        Block length. Defaults to ``floor(n ** (1/3))``.
    stat : callable, optional
        Statistic of the series. Defaults to the mean.
    B : int
        Bootstrap replicates.
    seed : int
        Seed.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``replicates``, ``block_len``,
        ``n_blocks``.

    References
    ----------
    Carlstein, E. (1986). The use of subseries values for estimating the
        variance of a general statistic from a stationary sequence. *Annals of
        Statistics*, 14(3), 1171-1179.

    Examples
    --------
    On dependent data the block bootstrap gives a materially larger standard
    error than the iid bootstrap, which is the entire point.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.zeros(600)
    >>> for i in range(1, 600):
    ...     x[i] = 0.9 * x[i - 1] + rng.normal()
    >>> blk = boot_nonoverlap_block(x, seed=1)["se"]
    >>> iid = float(np.std(x, ddof=1) / np.sqrt(600))
    >>> bool(blk > 2 * iid)
    True

    On independent data the two agree, so nothing is lost by using it.

    >>> z = rng.normal(size=600)
    >>> b2 = boot_nonoverlap_block(z, seed=1)["se"]
    >>> i2 = float(np.std(z, ddof=1) / np.sqrt(600))
    >>> bool(abs(b2 - i2) < 0.6 * i2)
    True

    The block count is reported, since it bounds resampling variability.

    >>> int(boot_nonoverlap_block(x, block_len=20, seed=1)["n_blocks"])
    30

    >>> boot_nonoverlap_block([1.0, 2.0], block_len=50)
    Traceback (most recent call last):
        ...
    ValueError: block_len must be between 1 and 2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    if block_len is None:
        block_len = max(int(n ** (1 / 3)), 1)
    block_len = int(block_len)
    if not 1 <= block_len <= n:
        raise ValueError(f"block_len must be between 1 and {n}")
    if stat is None:
        stat = np.mean
    n_blocks = n // block_len
    if n_blocks < 2:
        raise ValueError(
            f"block_len={block_len} leaves only {n_blocks} blocks; too few to resample"
        )
    blocks = np.array([x[i * block_len:(i + 1) * block_len] for i in range(n_blocks)])
    rng = np.random.default_rng(seed)
    reps = np.empty(int(B))
    for b in range(int(B)):
        pick = rng.integers(0, n_blocks, n_blocks)
        reps[b] = float(stat(blocks[pick].ravel()))
    est = float(stat(x))
    return RichResult(
        title="Non-overlapping block bootstrap",
        summary_lines=[("n", n), ("block", block_len), ("blocks", n_blocks),
                       ("se", float(np.std(reps, ddof=1)))],
        warnings=["non-overlapping blocks give only n/block_len resampling "
                  "units; the moving-block variant is more efficient but its "
                  "blocks are correlated"],
        payload={
            "estimate": est, "se": float(np.std(reps, ddof=1)),
            "ci": (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))),
            "replicates": reps, "block_len": block_len,
            "n_blocks": int(n_blocks), "B": int(B),
            "method": "boot_nonoverlap_block",
        },
    )


def cheatsheet():
    return "btnpb: iid bootstrap destroys dependence and understates SE; block length is the whole design choice"
