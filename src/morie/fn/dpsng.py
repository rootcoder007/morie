# morie.fn -- slice k04 (rootcoder007/morie)
"""Test the cluster count of a Dirichlet-process partition against its prior.

Source READ FROM THE CORPUS PDF: Ghosal, S. and van der Vaart, A.
(2017), *Fundamentals of Nonparametric Bayesian Inference*, section
4.1.5 "Number of Distinct Values", Proposition 4.8, which credits
Antoniak (1974), *Annals of Statistics* 2, 1152-1174.  (The Antoniak
paper itself was located as a scanned image PDF with no text layer and
could not be read; the Ghosal and van der Vaart statement is a primary
textbook source and is quoted here.)

Proposition 4.8: for an atomless base measure with total mass M, the
indicators D_i of "observation i is a new value" are INDEPENDENT
Bernoulli variables with

    P(D_i = 1) = M / (M + i - 1),      i = 1, ..., n

and K_n = sum_i D_i is the number of distinct values.  The proposition
gives the exact moments

    E(K_n)   = sum_{i=1}^{n}  M / (M + i - 1)
    var(K_n) = sum_{i=1}^{n}  M (i - 1) / (M + i - 1)^2

and states that (K_n - E K_n)/sd(K_n) is asymptotically standard normal,
with K_n also close in total variation to Poisson(E K_n).

Because the D_i are independent Bernoulli variables with known and
unequal probabilities, the exact null law of K_n is a Poisson-binomial
distribution, and this function convolves it directly rather than using
either approximation.  The convolution is a fixed n-step recursion --
no sampling, no tolerance, no early exit -- so the two-sided p-value is
exact and identical in both language arms.  The normal-approximation
deviate is reported alongside it for reference.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["dp_singularity_test"]


def _cluster_count(partition):
    """Number of distinct labels in a partition, and its length."""
    labels = list(partition)
    seen = []
    for a in labels:
        if a not in seen:
            seen.append(a)
    return len(seen), len(labels)


def _poisson_binomial_pmf(p):
    """Exact pmf of a sum of independent Bernoulli(p_i), by convolution."""
    pmf = [1.0]
    for pi in p:
        nxt = [0.0] * (len(pmf) + 1)
        for k, v in enumerate(pmf):
            nxt[k] += v * (1.0 - pi)
            nxt[k + 1] += v * pi
        pmf = nxt
    return pmf


def dp_singularity_test(partition, alpha=1.0, n=None):
    """Is the observed number of clusters consistent with DP concentration ``alpha``?

    Parameters
    ----------
    partition : array-like
        Cluster label per observation.  Only the number of distinct
        labels matters.  May instead be an int, in which case it is
        taken to be the cluster count K directly and ``n`` is required.
    alpha : float, default 1.0
        Dirichlet-process concentration M = |alpha| (total prior mass).
    n : int, optional
        Sample size; inferred from ``partition`` when it is a sequence.

    Returns
    -------
    RichResult
        keys: ``K`` (observed clusters), ``E_K``, ``var_K``, ``z``,
        ``p_value`` (exact two-sided Poisson-binomial), ``p_normal``
        (normal-approximation two-sided), ``alpha``, ``n``, ``method``.
    """
    if isinstance(partition, (int,)) and not isinstance(partition, bool):
        K = int(partition)
        if n is None:
            raise ValueError("n is required when partition is a bare cluster count")
        nn = int(n)
    else:
        K, nn = _cluster_count(partition)
        if n is not None:
            nn = int(n)
    M = float(alpha)
    if M <= 0.0:
        raise ValueError("alpha must be positive")
    if nn < 1:
        raise ValueError("n must be at least 1")
    if not 1 <= K <= nn:
        raise ValueError("cluster count must lie in 1..n")

    p = [M / (M + i - 1.0) for i in range(1, nn + 1)]
    e_k = sum(p)
    var_k = sum(M * (i - 1.0) / (M + i - 1.0) ** 2 for i in range(1, nn + 1))

    pmf = _poisson_binomial_pmf(p)
    # exact two-sided p-value: total mass of outcomes no more likely than
    # the one observed (the standard "method of small p" two-sided rule)
    pk = pmf[K]
    tol = 1e-12 * max(1.0, pk)
    p_exact = sum(v for v in pmf if v <= pk + tol)
    p_exact = min(1.0, max(0.0, p_exact))

    if var_k > 0.0:
        z = (K - e_k) / math.sqrt(var_k)
        # Abramowitz-Stegun-free: use the error function from the stdlib
        p_norm = min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))
    else:
        z = float("nan")
        p_norm = float("nan")

    return RichResult(
        payload={
            "K": K,
            "E_K": float(e_k),
            "var_K": float(var_k),
            "z": float(z),
            "p_value": float(p_exact),
            "p_normal": float(p_norm),
            "alpha": M,
            "n": nn,
            "method": "DP cluster-count test, exact Poisson-binomial (Ghosal and van der Vaart 2017 Prop. 4.8; Antoniak 1974)",
        }
    )


def cheatsheet():
    return "dpsng: DP partition cluster-count test against the prior"
