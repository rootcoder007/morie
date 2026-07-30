# morie.fn -- function file (rootcoder007/morie)
"""Entropy balancing."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["entropy_balancing"]


def entropy_balancing(X, treat, moments=1, max_iter=200, tol=1e-8):
    r"""Solve directly for weights that balance covariate moments exactly.

    Minimises the Kullback-Leibler divergence from uniform weights subject to
    exact moment constraints:

    .. math::
        \min_w \sum_i w_i \log \frac{w_i}{q_i}
        \quad\text{s.t.}\quad
        \sum_i w_i x_{ij} = \bar x_{1j}\;\forall j, \quad \sum_i w_i = 1.

    The distinction from propensity weighting is the direction of the logic.
    Propensity methods **model treatment assignment** and hope the resulting
    weights balance covariates, which is why they need iterating: fit, check
    balance, respecify, refit. Entropy balancing **imposes balance as a
    constraint** and finds the weights closest to uniform that satisfy it, so
    balance on the specified moments is exact by construction and there is
    nothing to check.

    What that does not buy is balance on anything unspecified. Constraining
    means leaves variances free; ``moments=2`` adds them. And exact balance on
    measured covariates says nothing at all about unmeasured ones -- the
    identifying assumption is untouched by any amount of balancing.

    Infeasibility is informative rather than a failure: if no weights satisfy
    the constraints, the treated group occupies a region of covariate space
    the controls do not, which is a positivity problem being surfaced.

    Parameters
    ----------
    X : array-like
        Covariates ``(n, p)``.
    treat : array-like
        Treatment indicator, 0/1.
    moments : int
        1 balances means; 2 adds variances.
    max_iter, tol
        Newton controls.

    Returns
    -------
    RichResult
        ``weights``, ``balance_achieved``, ``max_imbalance``, ``ess``,
        ``converged``.

    References
    ----------
    Hainmueller, J. (2012). Entropy balancing for causal effects.
        *Political Analysis*, 20(1), 25-46.

    Examples
    --------
    Balance on the specified moments is exact by construction, not
    approximate.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 3))
    >>> ps = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    >>> tr = (rng.random(600) < ps).astype(float)
    >>> r = entropy_balancing(X, tr)
    >>> bool(r["max_imbalance"] < 1e-6)
    True

    Constraining means leaves variances free, which is what moments=2 is for.

    >>> r2 = entropy_balancing(X, tr, moments=2)
    >>> bool(r2["max_imbalance"] < 1e-6)
    True
    >>> int(r2["n_constraints"]) > int(r["n_constraints"])
    True

    Weights stay positive and sum to one, so they are a reweighting rather
    than a subtraction.

    >>> bool(r["weights"].min() > 0 and abs(r["weights"].sum() - 1) < 1e-9)
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    if X.shape[0] != tr.size:
        raise ValueError(f"X has {X.shape[0]} rows but treat has {tr.size}")
    if not np.all((tr == 0) | (tr == 1)):
        raise ValueError("treat must be 0/1")
    moments = int(moments)
    if moments not in (1, 2):
        raise ValueError("moments must be 1 or 2")
    t1, t0 = tr == 1, tr == 0
    if not t1.any() or not t0.any():
        raise ValueError("both treatment groups must be non-empty")

    def design(A):
        cols = [A]
        if moments >= 2:
            cols.append(A**2)
        return np.column_stack(cols)

    C = design(X[t0])
    target = design(X[t1]).mean(axis=0)
    Cc = C - target                     # constraints become sum(w * Cc) = 0

    lam = np.zeros(Cc.shape[1])
    converged = False
    for _ in range(max_iter):
        z = Cc @ lam
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        g = w @ Cc
        if np.max(np.abs(g)) < tol:
            converged = True
            break
        Hm = (Cc * w[:, None]).T @ Cc - np.outer(g, g)
        try:
            step = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Hm, g, rcond=None)[0]
        lam = lam - step
    z = Cc @ lam
    z -= z.max()
    w = np.exp(z)
    w /= w.sum()
    imb = float(np.max(np.abs(w @ Cc)))
    ess = float(1.0 / np.sum(w**2))
    return RichResult(
        title="Entropy balancing",
        summary_lines=[("controls", int(t0.sum())), ("constraints", int(Cc.shape[1])),
                       ("max imbalance", imb), ("ESS", ess)],
        warnings=(["balance is exact only on the moments you specified, and "
                   "says nothing about unmeasured confounders"]
                  + ([] if converged else
                     ["the constraints could not be satisfied: the treated "
                      "group may occupy covariate regions the controls do not, "
                      "which is a positivity problem"])),
        payload={
            "weights": w, "lambda": lam,
            "balance_achieved": bool(imb < 1e-6), "max_imbalance": imb,
            "ess": ess, "n_constraints": int(Cc.shape[1]),
            "moments": moments, "converged": converged,
            "target": target, "method": "entropy_balancing",
        },
    )


def cheatsheet():
    return "ebalw: IMPOSES balance as a constraint instead of modelling assignment; infeasible = positivity problem"
