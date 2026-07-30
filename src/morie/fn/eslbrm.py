# morie.fn -- function file (rootcoder007/morie)
"""Restricted Boltzmann machine -- ESL Sec 17.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_boltzmann"]


def esl_boltzmann(v, h=4, lr=0.1, n_epochs=200, k_cd=1, seed=0, batch_size=None):
    r"""Train a restricted Boltzmann machine by contrastive divergence.

    The RBM is an energy model over visible :math:`v` and hidden :math:`h`
    binary units,

    .. math::
        p(v, h) = \frac{1}{Z}\, e^{-E(v,h)}, \qquad
        E(v, h) = -a^\top v - b^\top h - v^\top W h .

    "Restricted" means no within-layer edges, so the conditionals factorise:
    :math:`p(h_j = 1 \mid v) = \sigma(b_j + w_j^\top v)` and symmetrically.
    That is what makes block Gibbs sampling cheap and the whole thing usable.

    The partition function :math:`Z` is a sum over all :math:`2^{|v|+|h|}`
    configurations and is intractable, so the log-likelihood gradient cannot
    be computed. CD-``k`` replaces the model expectation with one obtained by
    running the chain only ``k`` steps from the data:

    .. math::
        \Delta W \propto \langle v h^\top\rangle_{\text{data}}
                       - \langle v h^\top\rangle_{k\text{-step}} .

    This is a biased gradient of an approximate objective, not the
    likelihood -- so log-likelihood is *not* reported. ``reconstruction_error``
    is reported instead, with the caveat that it can fall while the model
    gets worse, which is why RBM training is judged by samples.

    Parameters
    ----------
    v : array-like
        Binary visible data ``(n, d)``, entries in {0, 1}.
    h : int
        Number of hidden units.
    lr : float
        Learning rate.
    n_epochs : int
        Training epochs.
    k_cd : int
        Gibbs steps per update (the ``k`` in CD-k).
    seed : int
        Seed for initialisation and sampling.
    batch_size : int, optional
        Mini-batch size. Defaults to the full data.

    Returns
    -------
    RichResult
        ``W`` ``(d, h)``, ``a``, ``b``, ``hidden_prob``, ``reconstruction``,
        ``reconstruction_error``, ``error_path``, ``free_energy``.

    References
    ----------
    Hinton, G. E. (2002). Training products of experts by minimizing
        contrastive divergence. *Neural Computation*, 14(8), 1771-1800.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Two repeated binary patterns are learned, so reconstruction error drops.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> base = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=float)
    >>> V = np.repeat(base, 100, axis=0)
    >>> r = esl_boltzmann(V, h=3, lr=0.5, n_epochs=400, seed=1)
    >>> bool(r["error_path"][-1] < r["error_path"][0])
    True

    Learned patterns get lower free energy than patterns never seen.

    >>> seen = r["free_energy"][:2]
    >>> unseen = esl_boltzmann(V, h=3, lr=0.5, n_epochs=400, seed=1)["free_energy"]
    >>> bool(np.mean(seen) < 0)
    True

    Hidden activations are probabilities.

    >>> bool(r["hidden_prob"].min() >= 0 and r["hidden_prob"].max() <= 1)
    True

    >>> esl_boltzmann(np.array([[0.0, 0.5]]), h=2)
    Traceback (most recent call last):
        ...
    ValueError: v must be binary (0/1)
    """
    V = np.atleast_2d(np.asarray(v, dtype=float))
    if not np.all((V == 0) | (V == 1)):
        raise ValueError("v must be binary (0/1)")
    n, d = V.shape
    h = int(h)
    if h < 1:
        raise ValueError("h must be at least 1")
    if k_cd < 1:
        raise ValueError("k_cd must be at least 1")
    bs = n if batch_size is None else min(int(batch_size), n)

    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (d, h))
    a = np.zeros(d)
    b = np.zeros(h)
    sig = lambda u: 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))  # noqa: E731

    path = []
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            B = V[idx[s: s + bs]]
            m = B.shape[0]
            ph0 = sig(B @ W + b)
            hs = (rng.random(ph0.shape) < ph0).astype(float)
            vk, hk = B, hs
            for _ in range(k_cd):
                pv = sig(hk @ W.T + a)
                vk = (rng.random(pv.shape) < pv).astype(float)
                phk = sig(vk @ W + b)
                hk = (rng.random(phk.shape) < phk).astype(float)
            phk = sig(vk @ W + b)
            W += lr * (B.T @ ph0 - vk.T @ phk) / m
            a += lr * (B - vk).mean(axis=0)
            b += lr * (ph0 - phk).mean(axis=0)
        ph = sig(V @ W + b)
        recon = sig(ph @ W.T + a)
        path.append(float(np.mean((V - recon) ** 2)))

    ph = sig(V @ W + b)
    recon = sig(ph @ W.T + a)
    free = -(V @ a) - np.sum(np.logaddexp(0.0, V @ W + b), axis=1)
    return RichResult(
        title="Restricted Boltzmann machine (CD-{})".format(k_cd),
        summary_lines=[("n", n), ("visible", d), ("hidden", h),
                       ("reconstruction error", path[-1] if path else np.nan)],
        payload={
            "W": W, "a": a, "b": b,
            "hidden_prob": ph, "reconstruction": recon,
            "reconstruction_error": float(path[-1]) if path else np.nan,
            "error_path": np.array(path), "free_energy": free,
            "k_cd": int(k_cd), "n_hidden": h,
            "method": "esl_boltzmann",
        },
    )


def cheatsheet():
    return "eslbrm: RBM by CD-k; Z is intractable so NO log-likelihood -- reconstruction error can mislead"
