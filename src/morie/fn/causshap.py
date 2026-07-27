# morie.fn -- function file (rootcoder007/morie)
"""Shapley-value decomposition of a model's causal contributions."""

import itertools

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_shap_decomposition"]


def causal_shap_decomposition(value_fn, features, n_samples=None, seed=0):
    r"""Exact or sampled Shapley values for a coalitional value function.

    .. math:: \phi_i = \sum_{S \subseteq N \setminus \{i\}}
              \frac{|S|!\,(|N|-|S|-1)!}{|N|!}
              \big[v(S \cup \{i\}) - v(S)\big].

    Exact enumeration is used when the feature count is small
    (:math:`2^p` coalitions); otherwise the permutation sampling
    estimator averages marginal contributions over random orderings,
    which is unbiased. Efficiency -- the values summing to
    :math:`v(N) - v(\emptyset)` -- holds exactly in the enumerated case
    and in expectation when sampled, and is reported so the caller can
    check it.

    Parameters
    ----------
    value_fn : callable
        ``value_fn(subset) -> float`` where ``subset`` is a tuple of
        feature names. Must accept the empty tuple.
    features : sequence
        The feature names (the player set N).
    n_samples : int, optional
        Permutations to sample. None = exact enumeration (allowed up to
        12 features).
    seed : int, default 0
        RNG seed for the sampled variant.

    Returns
    -------
    RichResult
        keys: ``shapley`` (dict feature -> value), ``total``,
        ``grand_minus_empty``, ``efficiency_gap``, ``exact``,
        ``n_features``, ``method``.

    References
    ----------
    Shapley, L. S. (1953). A value for n-person games. In
    *Contributions to the Theory of Games II*, 307-317.

    Strumbelj, E. & Kononenko, I. (2014). Explaining prediction models
    and individual predictions with feature contributions. *Knowledge
    and Information Systems*, 41(3), 647-665. (the permutation
    sampling estimator)
    """
    if not callable(value_fn):
        raise ValueError("value_fn must be callable.")
    feats = list(features)
    p = len(feats)
    if p < 1:
        raise ValueError("need at least one feature.")

    v_empty = float(value_fn(()))
    v_all = float(value_fn(tuple(feats)))

    if n_samples is None:
        if p > 12:
            raise ValueError(f"exact enumeration needs p <= 12, got {p}; pass n_samples.")
        phi = dict.fromkeys(feats, 0.0)
        from math import factorial

        for i in feats:
            rest = [f for f in feats if f != i]
            for k in range(len(rest) + 1):
                w = factorial(k) * factorial(p - k - 1) / factorial(p)
                for S in itertools.combinations(rest, k):
                    phi[i] += w * (float(value_fn(tuple(S) + (i,))) - float(value_fn(S)))
        exact = True
    else:
        B = int(n_samples)
        if B < 1:
            raise ValueError(f"n_samples must be at least 1, got {B}.")
        rng = np.random.default_rng(seed)
        phi = dict.fromkeys(feats, 0.0)
        for _ in range(B):
            order = rng.permutation(p)
            prefix = []
            prev = v_empty
            for idx in order:
                f = feats[idx]
                prefix.append(f)
                cur = float(value_fn(tuple(prefix)))
                phi[f] += (cur - prev) / B
                prev = cur
        exact = False

    total = float(sum(phi.values()))
    return RichResult(
        payload={
            "shapley": phi,
            "total": total,
            "grand_minus_empty": v_all - v_empty,
            "efficiency_gap": total - (v_all - v_empty),
            "exact": exact,
            "n_features": p,
            "method": "Shapley decomposition (exact enumeration or permutation sampling)",
        }
    )


def cheatsheet():
    return "causshap: phi_i by weighted marginal contributions; sum = v(N) - v(empty)"
