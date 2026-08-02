# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Denoising diffusion probabilistic model (DDPM)."""

from . import _array_core as np

from ._richresult import RichResult
from .hmdfw import beta_schedule_values, lcg_normal

__all__ = ["geron_ddpm"]


def geron_ddpm(X, T=10, beta_schedule="linear", epochs=200, lr=0.05, seed=0):
    """
    Denoising diffusion probabilistic model (DDPM).

    Formula: x_t = sqrt(a_t)*x_0 + sqrt(1 - a_t)*eps; train eps_theta(x_t, t)

    Both halves are implemented. The forward process uses the schedule
    from :mod:`morie.fn.hmdfw`, so the ``alpha_bar`` used for training is
    the same object the forward/reverse modules use. The model
    ``eps_theta(x_t, t) = A_t x_t + b_t`` is a per-timestep affine map,
    trained by gradient descent on the DDPM objective

        L = E ||eps - eps_theta(x_t, t)||^2

    over deterministically drawn ``(x_0, eps)`` pairs.

    A per-timestep affine model is small enough to reason about and large
    enough to be right: the Bayes-optimal predictor of ``eps`` given
    ``x_t`` under a Gaussian data model *is* affine, so the trained
    coefficients can be compared with a closed form: the optimum is
    ``A_t = sqrt(1-abar_t) / (abar_t Var(x_0) + 1 - abar_t)`` times the
    data variance term. ``loss_by_t`` reports where the model is
    struggling; with a small deterministic training set that profile is
    driven by the particular noise draw, not only by ``t``.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Training data.
    T : int, default 10
        Diffusion steps.
    beta_schedule : {"linear", "cosine"} or array-like, default "linear"
    epochs : int, default 200
    lr : float, default 0.05
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: loss_history, final_loss, loss_by_t, A, b, alpha_bar,
        betas, sample, monotone, estimate, n, method.

    Examples
    --------
    Training reduces the objective, and the per-step losses are all
    finite and non-negative:

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> r = geron_ddpm(X, T=4, epochs=400, lr=0.1, seed=1)
    >>> r["final_loss"] < r["loss_history"][0]
    True
    >>> len(r["loss_by_t"]), len(r["A"]), len(r["alpha_bar"])
    (4, 4, 4)
    >>> min(r["loss_by_t"]) >= 0.0
    True

    Gradient descent on this objective is monotone at this step size:

    >>> r["monotone"]
    True

    A sample can be drawn from the trained model, and it has the shape of
    the data:

    >>> len(r["sample"]), len(r["sample"][0])
    (1, 1)

    References
    ----------
    Géron Ch 18
    """
    A0 = np.atleast_2d(np.asarray(X, dtype=float))
    if A0.ndim != 2 or A0.size == 0:
        raise ValueError(f"geron_ddpm: X must be a non-empty (m, d) array, got shape {A0.shape}")
    if not np.all(np.isfinite(A0)):
        raise ValueError("geron_ddpm: X contains non-finite values")
    Ti = int(T)
    if Ti < 1:
        raise ValueError(f"geron_ddpm: T must be >= 1, got {T!r}")
    betas = beta_schedule_values(Ti, beta_schedule)
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("geron_ddpm: every beta must lie strictly in (0, 1)")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_ddpm: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_ddpm: lr must be positive and finite, got {lr!r}")

    abar = np.cumprod(1.0 - betas)
    m, d = A0.shape

    # Fixed noise draws, one per (timestep, sample), so the objective is a
    # deterministic function of the parameters and gradient descent on it
    # is checkable.
    eps = np.stack([lcg_normal((m, d), seed + 1 + t) for t in range(Ti)])
    xt = np.stack([np.sqrt(abar[t]) * A0 + np.sqrt(1.0 - abar[t]) * eps[t] for t in range(Ti)])

    Aw = np.zeros(Ti)
    bw = np.zeros((Ti, d))
    hist = []
    for _ in range(E):
        total = 0.0
        for t in range(Ti):
            pred = Aw[t] * xt[t] + bw[t]
            diff = pred - eps[t]
            total += float(np.mean(diff**2))
            Aw[t] -= eta * float(np.mean(2.0 * diff * xt[t]))
            bw[t] -= eta * (2.0 * diff).mean(axis=0)
        hist.append(total / Ti)
        if not np.all(np.isfinite(Aw)) or not np.all(np.isfinite(bw)):
            raise ValueError("geron_ddpm: training diverged; lower lr")

    loss_by_t = []
    for t in range(Ti):
        pred = Aw[t] * xt[t] + bw[t]
        loss_by_t.append(float(np.mean((pred - eps[t]) ** 2)))
    final = float(np.mean(loss_by_t))

    # One ancestral sample with the trained model.
    x = lcg_normal((1, d), seed + 999)
    for t in range(Ti, 0, -1):
        e = Aw[t - 1] * x + bw[t - 1]
        b, a, ab = float(betas[t - 1]), float(1.0 - betas[t - 1]), float(abar[t - 1])
        mu = (x - (b / np.sqrt(1.0 - ab)) * e) / np.sqrt(a)
        x = mu + (np.sqrt(b) * lcg_normal((1, d), seed + 500 + t) if t > 1 else 0.0)

    mono = all(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))

    return RichResult(
        title="DDPM training",
        summary_lines=[("Steps", Ti), ("Final loss", final), ("Epochs", E)],
        interpretation="The objective is a plain regression on the injected noise, which is why DDPM training is so stable.",
        payload={
            "loss_history": hist,
            "final_loss": final,
            "loss_by_t": loss_by_t,
            "A": Aw.tolist(),
            "b": bw.tolist(),
            "alpha_bar": abar.tolist(),
            "betas": betas.tolist(),
            "sample": x.tolist(),
            "monotone": bool(mono),
            "T": Ti,
            "estimate": final,
            "n": int(m),
            "method": "DDPM with a per-timestep affine eps-model trained on the noise-prediction objective",
        },
    )


def cheatsheet():
    return "hmddpm: Denoising diffusion probabilistic model (DDPM)"
