# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet process -- stick-breaking construction."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_dirichlet_proc"]


def esl_dirichlet_proc(alpha=1.0, G0=None, n_atoms=50, size=None, seed=0):
    r"""Draw a random measure :math:`G \sim \mathrm{DP}(\alpha, G_0)`.

    Sethuraman's stick-breaking construction makes the draw explicit:

    .. math::
        \beta_k \sim \mathrm{Beta}(1, \alpha), \quad
        \pi_k = \beta_k \prod_{l<k}(1 - \beta_l), \quad
        \theta_k \sim G_0, \quad
        G = \sum_{k=1}^{\infty} \pi_k \delta_{\theta_k}.

    The draw is **discrete with probability one**, however continuous
    :math:`G_0` is -- which is the property that makes the DP a clustering
    prior: repeated draws from :math:`G` collide, and each distinct value is a
    cluster.

    :math:`\alpha` controls fragmentation. Small :math:`\alpha` puts nearly
    all mass on the first few atoms; the expected number of distinct values
    among :math:`n` draws grows like :math:`\alpha \log n`, so it grows with
    the data rather than being fixed in advance.

    Truncating at ``n_atoms`` leaves :math:`\prod_k (1-\beta_k)` of the stick
    unbroken. That residual is reported as ``truncation_mass``; if it is not
    negligible the truncation is distorting the draw and ``n_atoms`` needs
    raising.

    Parameters
    ----------
    alpha : float
        Concentration, positive.
    G0 : callable, optional
        Base measure sampler, ``G0(size, rng) -> atoms``. Defaults to
        standard normal.
    n_atoms : int
        Truncation level.
    size : int, optional
        If given, also draw this many observations from ``G``.
    seed : int
        Seed.

    Returns
    -------
    RichResult
        ``weights`` (summing to <= 1), ``atoms``, ``truncation_mass``,
        ``samples`` and ``n_clusters`` when ``size`` is given,
        ``expected_clusters``.

    References
    ----------
    Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
        *Statistica Sinica*, 4, 639-650.
    Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric
        problems. *Annals of Statistics*, 1(2), 209-230.

    Examples
    --------
    Weights are a probability vector once the truncation residual is added.

    >>> import numpy as np
    >>> r = esl_dirichlet_proc(alpha=2.0, n_atoms=200, seed=1)
    >>> bool(abs(r["weights"].sum() + r["truncation_mass"] - 1.0) < 1e-12)
    True

    Small alpha concentrates mass on the first atoms; large alpha spreads it.

    >>> lo = esl_dirichlet_proc(alpha=0.1, n_atoms=200, seed=1)["weights"][0]
    >>> hi = esl_dirichlet_proc(alpha=20.0, n_atoms=200, seed=1)["weights"][0]
    >>> bool(lo > hi)
    True

    Draws from ``G`` repeat, so the sample is discrete however continuous
    ``G0`` is -- and the cluster count tracks the alpha*log(n) rate.

    >>> d = esl_dirichlet_proc(alpha=2.0, n_atoms=400, size=500, seed=1)
    >>> bool(d["n_clusters"] < 500)
    True
    >>> bool(0.3 < d["n_clusters"] / d["expected_clusters"] < 3.0)
    True

    >>> esl_dirichlet_proc(alpha=0.0)
    Traceback (most recent call last):
        ...
    ValueError: alpha must be positive
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    n_atoms = int(n_atoms)
    if n_atoms < 1:
        raise ValueError("n_atoms must be at least 1")
    rng = np.random.default_rng(seed)

    betas = rng.beta(1.0, alpha, n_atoms)
    remain = np.r_[1.0, np.cumprod(1.0 - betas)[:-1]]
    weights = betas * remain
    trunc = float(np.prod(1.0 - betas))

    atoms = rng.normal(size=n_atoms) if G0 is None else np.asarray(G0(n_atoms, rng)).ravel()
    if atoms.size != n_atoms:
        raise ValueError(f"G0 returned {atoms.size} atoms, expected {n_atoms}")

    payload = {
        "weights": weights, "atoms": atoms,
        "truncation_mass": trunc, "alpha": float(alpha),
        "n_atoms": n_atoms,
        "method": "esl_dirichlet_proc",
    }
    warn = []
    if trunc > 1e-3:
        warn.append(
            f"{trunc:.3g} of the stick is unbroken at n_atoms={n_atoms}; "
            "the truncation is distorting the draw -- raise n_atoms"
        )
    if size is not None:
        size = int(size)
        p = weights / weights.sum()
        pick = rng.choice(n_atoms, size=size, p=p)
        payload["samples"] = atoms[pick]
        payload["labels"] = pick
        payload["n_clusters"] = int(np.unique(pick).size)
        payload["expected_clusters"] = float(alpha * np.log1p(size / alpha))
    return RichResult(
        title="Dirichlet process (stick-breaking)",
        summary_lines=[("alpha", float(alpha)), ("atoms", n_atoms),
                       ("truncation mass", trunc)],
        warnings=warn,
        payload=payload,
    )


def cheatsheet():
    return "esldai: stick-breaking DP; draws are DISCRETE a.s. -- check truncation_mass is negligible"
