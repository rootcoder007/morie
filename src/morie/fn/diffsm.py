# morie.fn -- function file (rootcoder007/morie)
"""Denoising score matching."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["diffusion_score_matching"]


def diffusion_score_matching(x, score, sigma=0.1, n_noise=16, seed=0):
    r"""Denoising score-matching objective at a single noise level.

    .. math::
        J(\theta) = \mathbb{E}_{x, \tilde x}
            \left\lVert s_\theta(\tilde x)
            + \frac{\tilde x - x}{\sigma^2}\right\rVert^2,
        \qquad \tilde x = x + \sigma\varepsilon .

    Vincent's identity is what makes this useful: minimising this is
    equivalent to matching the score of the *noised* data distribution, and it
    needs **no Jacobian trace** -- unlike plain score matching, whose trace
    term costs :math:`d` extra evaluations per point and makes high dimension
    unaffordable. The target :math:`-(\tilde x - x)/\sigma^2` is available in
    closed form because the noise was added deliberately.

    The bias this buys is explicit and is the whole design of diffusion
    models: the objective targets :math:`p_\sigma`, not :math:`p`. Small
    :math:`\sigma` keeps the bias small but leaves the score unconstrained
    where data is sparse; large :math:`\sigma` smooths the target into
    something easy to learn but far from the truth. Annealing across a
    schedule of :math:`\sigma` is how that tension is resolved.

    Parameters
    ----------
    x : array-like
        Clean samples ``(n, d)``.
    score : callable
        Model score ``s(x) -> (n, d)``.
    sigma : float
        Noise level, positive.
    n_noise : int
        Noise draws per sample.
    seed : int
        Seed.

    Returns
    -------
    RichResult
        ``objective``, ``sigma``, ``per_sample``, ``target_norm``.

    References
    ----------
    Vincent, P. (2011). A connection between score matching and denoising
        autoencoders. *Neural Computation*, 23(7), 1661-1674.

    Examples
    --------
    The objective is minimised at the score of the NOISED density, not of the
    data: for a standard normal with noise level sigma the optimal slope is
    ``1 / (1 + sigma^2)``, not 1. Enough noise draws are needed for the
    Monte Carlo estimate to resolve that -- at the default 16 the ordering
    among nearby candidates is not reliable.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 1))
    >>> cand = (0.25, 0.5, 1.0, 2.0, 4.0)
    >>> J = [diffusion_score_matching(X, lambda z, a=a: -a * z, sigma=0.5,
    ...                               n_noise=64)["objective"] for a in cand]
    >>> float(cand[int(np.argmin(J))])
    1.0

    The bias is real and grows with sigma: the theoretical optimum moves away
    from 1 as the noise level rises.

    >>> [float(round(1 / (1 + s ** 2), 3)) for s in (0.1, 0.5, 1.0)]
    [0.99, 0.8, 0.5]

    No Jacobian trace is needed, so the cost does not grow with dimension --
    the objective is finite in 50 dimensions at the same price.

    >>> Xd = rng.normal(size=(200, 50))
    >>> r = diffusion_score_matching(Xd, lambda z: -z, sigma=0.2)
    >>> bool(np.isfinite(r["objective"]))
    True

    >>> diffusion_score_matching(X, lambda z: -z, sigma=0.0)
    Traceback (most recent call last):
        ...
    ValueError: sigma must be positive
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    X = np.atleast_2d(np.asarray(x, dtype=float))
    n, d = X.shape
    rng = np.random.default_rng(seed)
    per = np.zeros(n)
    for _ in range(int(n_noise)):
        eps = rng.normal(size=X.shape)
        xt = X + sigma * eps
        s = np.atleast_2d(np.asarray(score(xt), dtype=float))
        if s.shape != X.shape:
            raise ValueError(f"score returned {s.shape}, expected {X.shape}")
        # Closed-form target: the noise was added deliberately, so no trace.
        target = -(xt - X) / sigma**2
        per += ((s - target) ** 2).sum(axis=1)
    per /= int(n_noise)
    return RichResult(
        title="Denoising score matching",
        summary_lines=[("n", n), ("d", d), ("sigma", sigma),
                       ("objective", float(per.mean()))],
        warnings=["the objective targets the NOISED distribution p_sigma, not "
                  "p; small sigma reduces the bias but leaves the score "
                  "unconstrained where data is sparse"],
        payload={
            "objective": float(per.mean()), "sigma": sigma,
            "per_sample": per, "target_norm": float(1.0 / sigma**2),
            "n_noise": int(n_noise), "method": "diffusion_score_matching",
        },
    )


def cheatsheet():
    return "diffsm: Vincent's identity removes the Jacobian trace; targets p_sigma not p -- anneal sigma"
