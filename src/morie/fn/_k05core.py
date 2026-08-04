"""Slice-local helpers for the k05 batch.

The Philox / AS 241 generator this batch needs now lives, working, in
``morie.fn._rng`` -- it had been broken by the de-numpy campaign, which is
why a transcription of the R arm sat here; the transcription is gone and
these are thin adapters that return plain lists.
"""

from ._rng import normal_quantile as _normal_quantile
from ._rng import philox4x32, random_normal, random_uniform

__all__ = []


def runif(n, seed=0, stream=0):
    """``n`` uniforms in the OPEN interval (0, 1), as a plain list."""
    return [float(v) for v in random_uniform(n, seed=seed, stream=stream)]


def qnorm(p):
    """Normal quantile, Wichura's AS 241 (the algorithm R's qnorm uses)."""
    p = float(p)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return float(_normal_quantile([p])[0])


def rnorm(n, seed=0, stream=0):
    """``n`` standard normals by inverse CDF, as a plain list."""
    return [float(v) for v in random_normal(n, seed=seed, stream=stream)]


def permutation(n, seed=0, stream=0):
    """A Fisher-Yates permutation of ``range(n)`` driven by ``runif``.

    Swaps run downward, i from n-1 to 1, consuming one uniform each, so
    the R mirror can reproduce it exactly by consuming the same stream
    in the same order.
    """
    idx = list(range(n))
    u = runif(max(n - 1, 0), seed=seed, stream=stream)
    for pos, i in enumerate(range(n - 1, 0, -1)):
        j = int(u[pos] * (i + 1))
        if j > i:
            j = i
        idx[i], idx[j] = idx[j], idx[i]
    return idx
