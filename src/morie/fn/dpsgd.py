# morie.fn -- function file (rootcoder007/morie)
"""DP-SGD with per-example clipping and noise."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["dp_sgd"]


def dp_sgd(grads, C=1.0, sigma=1.0, lr=0.1, theta=None, seed=None):
    r"""One DP-SGD step: clip each example's gradient, average, add noise.

    .. math::
        \tilde g = \frac{1}{B}\left(
            \sum_{i=1}^{B} g_i \min\!\left(1, \frac{C}{\lVert g_i\rVert_2}\right)
            + \mathcal{N}(0, \sigma^2 C^2 I)\right).

    Two details carry the guarantee and are both easy to get wrong.

    Clipping must be **per example**, before averaging. Clipping the averaged
    gradient bounds nothing about any individual's contribution and provides no
    privacy at all -- it is the single most common way a "DP-SGD"
    implementation turns out not to be private.

    The noise scales with :math:`C`, so raising the clipping bound to preserve
    signal also raises the noise proportionally. There is no free setting; the
    useful range is usually well below the median gradient norm, and
    ``clipped_fraction`` is returned so the bite is visible.

    This function performs one step. Accounting across steps is not free and
    is not done here -- compose with
    :func:`~morie.fn.dprnyi.renyi_dp_composition`, ideally with
    :func:`~morie.fn.dpamp.privacy_amplification` for the sampling.

    Parameters
    ----------
    grads : array-like
        Per-example gradients ``(B, p)``. **Not** an averaged gradient.
    C : float
        Per-example L2 clipping bound, positive.
    sigma : float
        Noise multiplier; noise sd is ``sigma * C``.
    lr : float
        Learning rate.
    theta : array-like, optional
        Current parameters; when given, the updated ones are returned.
    seed : int, optional
        Seed; leave ``None`` for real training.

    Returns
    -------
    RichResult
        ``update``, ``private_gradient``, ``clipped_fraction``,
        ``noise_sd``, ``theta``.

    References
    ----------
    Abadi, M., Chu, A., Goodfellow, I., et al. (2016). Deep learning with
        differential privacy. *CCS 2016*, 308-318.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-407.

    Examples
    --------
    Clipping is per example: a single enormous gradient cannot dominate the
    average, which is exactly the guarantee.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> G = rng.normal(size=(64, 5)) * 0.1
    >>> G_bad = G.copy(); G_bad[0] = 1e6                  # one runaway example
    >>> a = dp_sgd(G, C=1.0, sigma=0.0, lr=0.1, seed=0)["private_gradient"]
    >>> b = dp_sgd(G_bad, C=1.0, sigma=0.0, lr=0.1, seed=0)["private_gradient"]
    >>> bool(np.linalg.norm(b - a) < 1.0 / 64 * 2)
    True

    Noise standard deviation is sigma times C, so a larger clip means more
    noise.

    >>> float(dp_sgd(G, C=2.0, sigma=1.5, seed=0)["noise_sd"])
    3.0

    The clipped fraction is reported so the bound's bite is visible.

    >>> bool(dp_sgd(G_bad, C=1.0, sigma=0.0, seed=0)["clipped_fraction"] > 0)
    True

    An averaged gradient is refused, since clipping it would guarantee
    nothing.

    >>> dp_sgd([0.1, 0.2, 0.3], C=1.0)
    Traceback (most recent call last):
        ...
    ValueError: grads must be per-example, shape (B, p); clipping an averaged gradient provides no privacy
    """
    C = float(C)
    if C <= 0:
        raise ValueError("C must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    G = np.asarray(grads, dtype=float)
    if G.ndim != 2:
        raise ValueError(
            "grads must be per-example, shape (B, p); clipping an averaged "
            "gradient provides no privacy"
        )
    B, p = G.shape
    norms = np.linalg.norm(G, axis=1)
    factor = np.minimum(1.0, C / np.maximum(norms, 1e-12))
    Gc = G * factor[:, None]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma * C, p) if sigma > 0 else np.zeros(p)
    gbar = (Gc.sum(axis=0) + noise) / B
    update = -lr * gbar
    payload = {
        "update": update, "private_gradient": gbar,
        "clipped_fraction": float(np.mean(norms > C)),
        "noise_sd": float(sigma * C), "C": C, "sigma": float(sigma),
        "batch_size": int(B), "method": "dp_sgd",
    }
    if theta is not None:
        th = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
        if th.size != p:
            raise ValueError(f"theta has {th.size} entries but gradients have {p}")
        payload["theta"] = th + update
    return RichResult(
        title="DP-SGD step",
        summary_lines=[("batch", int(B)), ("C", C), ("noise sd", float(sigma * C)),
                       ("clipped", payload["clipped_fraction"])],
        warnings=["this accounts for ONE step; compose across steps with "
                  "renyi_dp_composition, and apply subsampling amplification"],
        payload=payload,
    )


def cheatsheet():
    return "dpsgd: clip PER EXAMPLE before averaging -- clipping the mean gradient gives no privacy at all"
