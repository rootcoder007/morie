# morie.fn -- function file (rootcoder007/morie)
"""Expected a posteriori ability estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["eap_theta_estimator"]


def eap_theta_estimator(y, a=None, b=None, c=None, prior=(0.0, 1.0),
                        n_nodes=61):
    r"""Expected a posteriori estimate of :math:`\theta` -- the
    posterior MEAN rather than the mode,

    .. math:: \hat\theta_{EAP} = \frac{\sum_q X_q L(X_q) W_q}
                                      {\sum_q L(X_q) W_q},

    computed by Gauss-Hermite quadrature against the normal prior
    (Bock and Mislevy 1982, Eqs. (6)-(7)).

    Three properties distinguish it from MAP and all are tested.
    EAP requires NO optimisation -- it is two weighted sums, so it
    cannot fail to converge and has no multimodality problem, which
    is why it is the standard choice in adaptive testing. It is the
    minimum-mean-squared-error estimator under the prior, so it
    shrinks MORE than MAP toward the prior mean. And its posterior
    standard deviation is a genuine posterior spread, not a
    curvature approximation -- for skewed posteriors (short tests,
    extreme patterns) the two differ materially.

    The quadrature node count is the only tuning parameter and it is
    reported, because too few nodes silently biases the tails where
    extreme patterns live.

    Parameters
    ----------
    y : array-like of 0/1
        Item responses.
    a, b, c : array-like, optional
        Item parameters; ``b`` required.
    prior : (float, float)
        Normal prior mean and standard deviation.
    n_nodes : int, default 61
        Gauss-Hermite nodes.

    Returns
    -------
    RichResult
        keys: ``theta``, ``se`` (posterior SD), ``posterior_sd``,
        ``prior_mean``, ``prior_sd``, ``n_nodes``,
        ``shrinkage_vs_map``, ``no_optimisation`` (True),
        ``n_items``, ``method``.

    References
    ----------
    Bock, R. D. and Mislevy, R. J. (1982), "Adaptive EAP estimation
    of ability in a microcomputer environment", *Applied
    Psychological Measurement* 6:431-444, Eqs. (6)-(7).
    """
    from ._psycho import gauss_hermite, logistic_3pl

    yv = np.asarray(y, dtype=float).ravel()
    m = yv.size
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("responses must be binary 0/1.")
    if b is None:
        raise ValueError("item difficulties b are required.")
    bv = np.asarray(b, dtype=float).ravel()
    av = np.ones(m) if a is None else np.asarray(a, dtype=float).ravel()
    cv = np.zeros(m) if c is None else np.asarray(c, dtype=float).ravel()
    if not (bv.size == av.size == cv.size == m):
        raise ValueError("a, b, c must each have one entry per item.")
    mu, sd = float(prior[0]), float(prior[1])
    if sd <= 0:
        raise ValueError(f"the prior standard deviation must be positive, "
                         f"got {sd}.")
    nn = int(n_nodes)
    if nn < 5:
        raise ValueError(f"need at least 5 quadrature nodes, got {nn}.")
    X, W = gauss_hermite(nn, mu, sd)
    P = np.clip(logistic_3pl(X, av, bv, cv), 1e-12, 1 - 1e-12)
    ll = (yv * np.log(P) + (1 - yv) * np.log(1 - P)).sum(axis=1)
    ll = ll - ll.max()
    post = np.exp(ll) * W
    tot = float(post.sum())
    if tot <= 0:
        raise ValueError("the posterior mass underflowed; widen the prior "
                         "or add nodes.")
    post = post / tot
    th = float(np.sum(X * post))
    var = float(np.sum((X - th) ** 2 * post))
    shrink = None
    try:
        from .mapth import map_theta_estimator
        mp = map_theta_estimator(y, a=av, b=bv, c=cv, prior=prior)
        shrink = float(th - mp["theta"])
    except Exception:
        pass
    return RichResult(payload={
        "theta": th, "se": float(np.sqrt(var)),
        "posterior_sd": float(np.sqrt(var)),
        "prior_mean": mu, "prior_sd": sd, "n_nodes": nn,
        "shrinkage_vs_map": shrink,
        "no_optimisation": True,
        "why_no_optimisation": "EAP is two weighted sums, so it cannot fail "
                               "to converge and has no multimodality "
                               "problem -- the reason adaptive testing "
                               "prefers it",
        "se_note": "a genuine posterior standard deviation, not a curvature "
                   "approximation; the two differ materially for skewed "
                   "posteriors",
        "node_note": "too few nodes silently biases the tails, where "
                     "extreme patterns live",
        "n_items": int(m),
        "method": "EAP theta by Gauss-Hermite quadrature (Bock-Mislevy 1982)"})


def cheatsheet():
    return "eapth: posterior MEAN by quadrature -- no optimisation, shrinks more than MAP"
