# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variational autoencoder with latent Gaussian prior."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_vae", "vae_loss_and_grads"]


def _lcg(shape, seed, scale=0.1):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def _lcg_normal(shape, seed):
    """Box-Muller normal draws from the LCG stream (no global RNG state)."""
    n = int(np.prod(shape))
    m = n + (n % 2)
    s = int(seed) % 2**32
    u = np.empty(m)
    for i in range(m):
        s = (1664525 * s + 1013904223) % 2**32
        u[i] = (s + 0.5) / 2**32
    a, b = u[0::2], u[1::2]
    z = np.concatenate([np.sqrt(-2 * np.log(a)) * np.cos(2 * np.pi * b), np.sqrt(-2 * np.log(a)) * np.sin(2 * np.pi * b)])
    return z[:n].reshape(shape)


def vae_loss_and_grads(X, params, eps, beta=1.0):
    """ELBO (as a minimisation loss) and its exact gradients for fixed noise.

    ``params`` is ``(W_mu, b_mu, W_lv, b_lv, W_d, b_d)``. Splitting this out
    keeps the reparameterisation gradient checkable against finite
    differences with the noise held fixed.
    """
    Wmu, bmu, Wlv, blv, Wd, bd = params
    n, d = X.shape
    mu = X @ Wmu + bmu
    # Log-variance clamped: exp(lv) overflows long before the model is useful,
    # and a clamp is what every practical VAE implementation uses.
    lv = np.clip(X @ Wlv + blv, -30.0, 30.0)
    sd = np.exp(0.5 * lv)
    z = mu + eps * sd
    xhat = z @ Wd + bd
    diff = xhat - X
    recon = float(np.mean(diff * diff))
    kl = float(np.sum(0.5 * (mu * mu + np.exp(lv) - 1.0 - lv)) / n)
    loss = recon + beta * kl

    dxhat = 2.0 * diff / (n * d)
    dWd = z.T @ dxhat
    dbd = dxhat.sum(axis=0)
    dz = dxhat @ Wd.T
    dmu = dz + beta * mu / n
    dlv = dz * eps * 0.5 * sd + beta * 0.5 * (np.exp(lv) - 1.0) / n
    grads = (X.T @ dmu, dmu.sum(axis=0), X.T @ dlv, dlv.sum(axis=0), dWd, dbd)
    return loss, recon, kl, grads


def geron_vae(X, latent_dim=2, epochs=200, lr=0.05, beta=1.0, seed=0):
    """
    Variational autoencoder with latent Gaussian prior.

    Formula: encoder outputs mu, log_sigma; sample z; decoder reconstructs

    Trained by gradient descent on the negative ELBO, with the two pieces
    kept separate in the payload:

    * reconstruction ``mean ||x_hat - x||^2``;
    * ``KL(q(z|x) || N(0, I)) = 0.5 * sum(mu^2 + sigma^2 - 1 - log sigma^2)``,
      which has this closed form only because both distributions are
      Gaussian -- no sampling needed for that term.

    The **reparameterisation trick** is what makes the sampling
    differentiable: ``z = mu + sigma * eps`` moves the randomness into
    ``eps``, so the gradient flows through mu and sigma. The noise comes
    from a deterministic LCG, so a run is reproducible.

    Parameters
    ----------
    X : array-like
        Training data (n, d).
    latent_dim : int, default 2
        Latent width (1 <= latent_dim).
    epochs : int, default 200
        Gradient steps (>= 1).
    lr : float, default 0.05
        Learning rate (> 0).
    beta : float, default 1.0
        Weight on the KL term (beta-VAE); >= 0.
    seed : int, default 0
        LCG seed for weights and noise.

    Returns
    -------
    result : RichResult
        Keys: mu, log_var, z, reconstruction, recon_error, kl, elbo,
        loss_curve, estimate, n, method.

    Examples
    --------
    >>> X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    >>> r = geron_vae(X, latent_dim=1, epochs=300, lr=0.05)
    >>> r["mu"].shape, r["z"].shape
    ((5, 1), (5, 1))
    >>> bool(r["loss_curve"][-1] < r["loss_curve"][0])
    True
    >>> bool(r["kl"] >= 0.0)
    True

    A larger beta buys a tighter latent (smaller KL) at the cost of
    reconstruction -- the beta-VAE trade-off, measured:

    >>> r2 = geron_vae(X, latent_dim=1, epochs=300, lr=0.005, beta=20.0)
    >>> bool(r2["kl"] < r["kl"])
    True

    References
    ----------
    Géron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_vae: X must be a non-empty (n, d) matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_vae: X contains non-finite values")
    k = int(latent_dim)
    if k < 1:
        raise ValueError(f"geron_vae: latent_dim must be >= 1, got {k}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_vae: epochs must be >= 1, got {E}")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_vae: lr must be positive and finite, got {step}")
    b = float(beta)
    if not np.isfinite(b) or b < 0:
        raise ValueError(f"geron_vae: beta must be non-negative and finite, got {b}")

    n, d = A.shape
    params = [
        _lcg((d, k), int(seed) + 1),
        np.zeros(k),
        _lcg((d, k), int(seed) + 2),
        np.zeros(k),
        _lcg((k, d), int(seed) + 3),
        np.zeros(d),
    ]
    losses = []
    for e in range(E):
        eps = _lcg_normal((n, k), int(seed) + 1000 + e)
        loss, recon, kl, grads = vae_loss_and_grads(A, params, eps, b)
        if not np.isfinite(loss):
            raise ValueError(f"geron_vae: the loss became non-finite at epoch {e}; lower lr")
        losses.append(loss)
        for i in range(6):
            params[i] = params[i] - step * grads[i]
        if not np.all(np.isfinite(params[0])):
            raise ValueError(f"geron_vae: parameters diverged at epoch {e}; lower lr")

    Wmu, bmu, Wlv, blv, Wd, bd = params
    mu = A @ Wmu + bmu
    lv = np.clip(A @ Wlv + blv, -30.0, 30.0)
    eps = _lcg_normal((n, k), int(seed) + 999)
    z = mu + eps * np.exp(0.5 * lv)
    xhat = z @ Wd + bd
    recon = float(np.mean((xhat - A) ** 2))
    kl = float(np.sum(0.5 * (mu * mu + np.exp(lv) - 1.0 - lv)) / n)

    return RichResult(
        title="Variational autoencoder",
        summary_lines=[
            ("Latent dim", k),
            ("Reconstruction MSE", recon),
            ("KL to N(0, I)", kl),
            ("beta", b),
        ],
        interpretation=(
            "The KL term pulls every posterior towards the prior, which is what makes the latent space "
            "continuous enough to sample from; push beta too high and the decoder ignores z entirely."
        ),
        payload={
            "mu": mu,
            "log_var": lv,
            "z": z,
            "reconstruction": xhat,
            "recon_error": recon,
            "kl": kl,
            "elbo": -(recon + b * kl),
            "loss_curve": np.asarray(losses, dtype=float),
            "params": params,
            "beta": b,
            "estimate": recon + b * kl,
            "n": int(n),
            "method": "Gaussian VAE trained on the negative ELBO with the reparameterisation trick",
        },
    )


def cheatsheet():
    return "hmvae: Variational autoencoder with latent Gaussian prior"


# compact alias per ledger/NAMING.md
geronvae = geron_vae
