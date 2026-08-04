# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Noise Conditional Score Network (NCSN)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ncsn"]


def _lcg_normal(n, seed):
    """Box-Muller normals from the integer LCG: identical on every machine."""
    s = int(seed) % 2**32
    out = np.empty(n)
    i = 0
    while i < n:
        s = (1664525 * s + 1013904223) % 2**32
        u1 = (s + 0.5) / 2**32
        s = (1664525 * s + 1013904223) % 2**32
        u2 = (s + 0.5) / 2**32
        r = np.sqrt(-2.0 * np.log(u1))
        out[i] = r * np.cos(2.0 * np.pi * u2)
        if i + 1 < n:
            out[i + 1] = r * np.sin(2.0 * np.pi * u2)
        i += 2
    return out


def geron_ncsn(X, sigmas=(1.0,), epochs=400, lr=0.5, n_noise=32, seed=0,
               n_samples=0, langevin_steps=20, step_eps=0.05):
    """
    Noise Conditional Score Network (NCSN): a score-based generative model.

    Formula: train s_theta(x, sigma) to match grad_x log p_sigma(x)

    The score, grad_x log p(x), is learnable without ever computing the
    normalising constant -- that is the whole appeal, and denoising score
    matching is the trick that makes it possible: corrupt x by sigma*z
    and regress the network onto -z/sigma, whose expectation IS the score
    of the smoothed density. No integral, no partition function.

    One noise level is not enough. Where the data are sparse the score is
    both unlearned and useless for sampling, so NCSN trains a LADDER of
    sigmas at once: the large ones fill the space and give Langevin
    dynamics a gradient to follow from anywhere, the small ones sharpen
    the result. Annealed Langevin sampling walks down that ladder.

    The score model here is affine in x, one per noise level, which is
    exact for Gaussian data and is the case where the answer is known:
    the optimum is -(S + sigma^2 I)^-1 (x - mu), returned as ``analytic``
    beside the fit so the training can be CHECKED rather than trusted.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Training data.
    sigmas : sequence of float, default (1.0,)
        Noise ladder, largest first by convention (sorted here).
    epochs : int, default 400
    lr : float, default 0.1
        Dimensionless step in (0, 2): the true step is this divided by the
        noise level's curvature, so one value is stable across the ladder.
    n_noise : int, default 32
        Noise draws per training point.
    seed : int, default 0
    n_samples : int, default 0
        Annealed Langevin samples to draw.
    langevin_steps : int, default 20
        Steps per noise level.
    step_eps : float, default 0.05
        Base Langevin step size.

    Returns
    -------
    result : RichResult
        Keys: models, analytic, max_deviation, loss_history, samples,
        sigmas, estimate, n, method.

    Examples
    --------
    Data with mean 0 and variance 2 at sigma = 1: the exact score slope
    is -1/(2 + 1).

    >>> X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    >>> r = geron_ncsn(X, [1.0], epochs=600)
    >>> round(float(r["analytic"][0]["W"][0, 0]), 6)
    -0.333333
    >>> bool(abs(float(r["models"][0]["W"][0, 0]) + 1.0 / 3.0) < 0.05)
    True

    The denoising loss falls over training:

    >>> bool(r["loss_history"][0][-1] < r["loss_history"][0][0])
    True

    A larger sigma flattens the score, which is what makes the far tails
    navigable:

    >>> r2 = geron_ncsn(X, [1.0, 4.0], epochs=600)
    >>> a = {m["sigma"]: float(m["W"][0, 0]) for m in r2["analytic"]}
    >>> round(a[4.0], 6), round(a[1.0], 6)
    (-0.055556, -0.333333)
    >>> bool(a[4.0] > a[1.0])
    True

    Annealed Langevin sampling returns the requested number of points:

    >>> geron_ncsn(X, [1.0], epochs=100, n_samples=3)["samples"].shape
    (3, 1)

    References
    ----------
    Geron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_ncsn: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_ncsn: X contains non-finite values")
    sg = np.atleast_1d(np.asarray(sigmas, dtype=float)).ravel()
    if sg.size == 0:
        raise ValueError("geron_ncsn: sigmas is empty")
    if np.any(sg <= 0) or not np.all(np.isfinite(sg)):
        raise ValueError(f"geron_ncsn: every sigma must be positive and finite, got {sigmas!r}")
    sg = np.sort(sg)[::-1]  # large to small: the ladder is walked downward
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_ncsn: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not (0.0 < eta < 2.0):
        raise ValueError(f"geron_ncsn: lr is a dimensionless step and must lie in (0, 2), got {lr!r}")
    K = int(n_noise)
    if K < 1:
        raise ValueError(f"geron_ncsn: n_noise must be >= 1, got {n_noise!r}")
    S = int(n_samples)
    if S < 0:
        raise ValueError(f"geron_ncsn: n_samples must be >= 0, got {n_samples!r}")
    T = int(langevin_steps)
    if T < 1:
        raise ValueError(f"geron_ncsn: langevin_steps must be >= 1, got {langevin_steps!r}")

    m, d = A.shape
    mu = A.mean(axis=0)
    Xc = A - mu
    Cov = (Xc.T @ Xc) / m

    models, analytic, hists = [], [], []
    for si, sigma in enumerate(sg):
        Z = _lcg_normal(m * K * d, seed + 7919 * si + 1).reshape(m * K, d)
        base = np.repeat(A, K, axis=0)
        Xt = base + sigma * Z
        target = -Z / sigma  # score of the perturbation kernel
        W = np.zeros((d, d))
        b = np.zeros(d)
        hist = []
        N = Xt.shape[0]
        # The curvature of this level's loss is sigma^2 * E[[x,1][x,1]^T], which
        # grows like sigma^4; dividing it out makes one dimensionless lr stable
        # across the whole ladder instead of diverging at the large sigmas.
        aug = np.hstack([Xt, np.ones((N, 1))])
        lam = float(np.max(np.linalg.eigvalsh((aug.T @ aug) / N)))
        eta_eff = eta / (sigma**2 * max(lam, 1e-12))
        for _ in range(E):
            resid = Xt @ W.T + b - target
            loss = float(0.5 * sigma**2 * np.mean(np.sum(resid**2, axis=1)))
            hist.append(loss)
            gW = (sigma**2 / N) * (resid.T @ Xt)
            gb = (sigma**2 / N) * resid.sum(axis=0)
            W -= eta_eff * gW
            b -= eta_eff * gb
        resid = Xt @ W.T + b - target
        hist.append(float(0.5 * sigma**2 * np.mean(np.sum(resid**2, axis=1))))
        models.append({"sigma": float(sigma), "W": W, "b": b})
        Wa = -np.linalg.inv(Cov + sigma**2 * np.eye(d))
        analytic.append({"sigma": float(sigma), "W": Wa, "b": -Wa @ mu})
        hists.append(hist)

    dev = max(float(np.max(np.abs(mm["W"] - aa["W"]))) for mm, aa in zip(models, analytic))

    samples = np.empty((0, d))
    if S > 0:
        s_rng = seed + 991
        x = mu + sg[0] * _lcg_normal(S * d, s_rng).reshape(S, d)
        sig_min = float(sg[-1])
        for si, sigma in enumerate(sg):
            alpha = float(step_eps) * (sigma**2) / (sig_min**2)
            W, b = models[si]["W"], models[si]["b"]
            for t in range(T):
                score = x @ W.T + b
                noise = _lcg_normal(S * d, s_rng + 7919 * si + 13 * t + 3).reshape(S, d)
                x = x + 0.5 * alpha * score + np.sqrt(alpha) * noise
        samples = x

    return RichResult(
        title="Noise conditional score network",
        summary_lines=[("Noise levels", int(sg.size)), ("Max deviation from exact score", dev), ("Samples", int(S))],
        interpretation="No partition function is ever needed; the ladder of sigmas is what makes Langevin sampling mix.",
        payload={
            "models": models,
            "analytic": analytic,
            "max_deviation": dev,
            "loss_history": hists,
            "samples": samples,
            "sigmas": sg,
            "mean": mu,
            "covariance": Cov,
            "estimate": models,
            "n": int(m),
            "method": "NCSN: affine score models fitted by denoising score matching, with annealed Langevin sampling",
        },
    )


def cheatsheet():
    return "hmncsn: Noise conditional score network by denoising score matching"


# compact alias per ledger/NAMING.md
geronncsn = geron_ncsn
