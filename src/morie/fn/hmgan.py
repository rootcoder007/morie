# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generative adversarial network: generator vs discriminator minimax."""

from . import _array_core as np

from ._richresult import RichResult
from .hmdfw import lcg_normal

__all__ = ["geron_gan"]


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def geron_gan(X, G=None, D=None, z_dim=1, epochs=200, lr=0.05, seed=0, non_saturating=True):
    """
    Generative adversarial network: generator vs discriminator minimax.

    Formula: min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]

    A working minimax game with linear players: the generator is
    ``G(z) = W_g z + b_g`` and the discriminator a logistic model
    ``D(x) = sigmoid(w_d . x + b_d)``. Both gradients are analytic, and
    the two are updated in alternation, discriminator first, which is the
    order the objective's inner maximum requires.

    The value function ``V = E[log D(x)] + E[log(1 - D(G(z)))]`` is
    tracked every epoch. Its one exactly known point is the equilibrium: a
    discriminator that outputs 0.5 everywhere gives ``V = 2 log 0.5 =
    -1.3863``, and that number is reported as ``equilibrium_value`` so
    "has it converged?" has a reference rather than a vibe.

    ``non_saturating`` switches the generator to the
    ``-log D(G(z))`` loss Goodfellow recommends: the minimax generator
    gradient vanishes exactly when the discriminator is winning, which is
    the failure mode the non-saturating form exists to avoid.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Real data.
    G : sequence, optional
        ``(W_g, b_g)`` with ``W_g`` of shape ``(z_dim, d)``.
    D : sequence, optional
        ``(w_d, b_d)`` with ``w_d`` of length ``d``.
    z_dim : int, default 1
    epochs : int, default 200
    lr : float, default 0.05
    seed : int, default 0
    non_saturating : bool, default True

    Returns
    -------
    result : RichResult
        Keys: G, D, value_history, d_loss, g_loss, samples, real_scores,
        fake_scores, equilibrium_value, mean_gap, estimate, n, method.

    Examples
    --------
    A perfectly confused discriminator scores the equilibrium value
    ``2 log 0.5``:

    >>> import math
    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> r0 = geron_gan(X, G=([[0.0]], [1.5]), D=([0.0], 0.0), epochs=1, lr=0.0)
    >>> round(r0["value_history"][0], 6) == round(2 * math.log(0.5), 6)
    True
    >>> round(r0["equilibrium_value"], 6)
    -1.386294

    Training moves the generated samples towards the real mean, so the
    gap between the two shrinks:

    >>> r = geron_gan(X, epochs=800, lr=0.1, seed=2)
    >>> r["mean_gap"] < r["initial_mean_gap"]
    True
    >>> len(r["samples"]), len(r["value_history"])
    (4, 800)

    A discriminator that is certain gives the generator no gradient under
    the minimax loss, but plenty under the non-saturating one:

    >>> a = geron_gan(X, G=([[0.0]], [-50.0]), D=([1.0], 0.0), epochs=1, lr=0.1,
    ...               non_saturating=False)
    >>> b = geron_gan(X, G=([[0.0]], [-50.0]), D=([1.0], 0.0), epochs=1, lr=0.1,
    ...               non_saturating=True)
    >>> round(a["g_grad_norm"], 12)
    0.0
    >>> b["g_grad_norm"] > 0.9
    True

    References
    ----------
    Géron Ch 18
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_gan: X must be a non-empty (m, d) array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_gan: X contains non-finite values")
    m, d = A.shape
    k = int(z_dim)
    if k < 1:
        raise ValueError(f"geron_gan: z_dim must be >= 1, got {z_dim!r}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_gan: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError(f"geron_gan: lr must be non-negative and finite, got {lr!r}")

    if G is None:
        Wg = np.zeros((k, d))
        bg = np.zeros(d)
    else:
        if len(G) != 2:
            raise ValueError("geron_gan: G must be (W_g, b_g)")
        Wg = np.atleast_2d(np.asarray(G[0], dtype=float))
        bg = np.atleast_1d(np.asarray(G[1], dtype=float))
        if Wg.shape != (k, d) or bg.size != d:
            raise ValueError(f"geron_gan: G must have W_g of shape {(k, d)} and b_g of length {d}")
    if D is None:
        wd = np.zeros(d)
        bd = 0.0
    else:
        if len(D) != 2:
            raise ValueError("geron_gan: D must be (w_d, b_d)")
        wd = np.atleast_1d(np.asarray(D[0], dtype=float))
        bd = float(D[1])
        if wd.size != d:
            raise ValueError(f"geron_gan: w_d must have length {d}, got {wd.size}")

    Z = lcg_normal((m, k), seed + 1)
    initial_gap = float(abs(np.mean(Z @ Wg + bg) - np.mean(A)))

    vals, dls, gls = [], [], []
    g_grad_norm = 0.0
    for _ in range(E):
        fake = Z @ Wg + bg
        d_real = _sigmoid(A @ wd + bd)
        d_fake = _sigmoid(fake @ wd + bd)
        v = float(np.mean(np.log(np.clip(d_real, 1e-12, 1.0))) + np.mean(np.log(np.clip(1 - d_fake, 1e-12, 1.0))))
        vals.append(v)
        dls.append(-v)

        # Discriminator ascends V.
        gw = (A * (1 - d_real)[:, None]).mean(axis=0) - (fake * d_fake[:, None]).mean(axis=0)
        gb = float(np.mean(1 - d_real) - np.mean(d_fake))
        wd = wd + eta * gw
        bd = bd + eta * gb

        # Generator: minimax dL/dfake = -d_fake * w, non-saturating = (d_fake - 1) * w.
        fake = Z @ Wg + bg
        d_fake = _sigmoid(fake @ wd + bd)
        coeff = (d_fake - 1.0) if non_saturating else (-d_fake)
        dfake = (coeff[:, None] * wd[None, :]) / m
        gWg = Z.T @ dfake
        gbg = dfake.sum(axis=0)
        g_grad_norm = float(np.linalg.norm(np.concatenate([gWg.ravel(), gbg])))
        gls.append(float(np.mean(-np.log(np.clip(d_fake, 1e-12, 1.0)))))
        Wg = Wg - eta * gWg
        bg = bg - eta * gbg

    fake = Z @ Wg + bg
    d_real = _sigmoid(A @ wd + bd)
    d_fake = _sigmoid(fake @ wd + bd)
    gap = float(abs(np.mean(fake) - np.mean(A)))

    return RichResult(
        title="GAN minimax training",
        summary_lines=[("Value", vals[-1]), ("Equilibrium", float(2 * np.log(0.5))), ("Mean gap", gap)],
        interpretation="At equilibrium D(x) = 0.5 everywhere and V = 2 log 0.5; distance from that is the honest progress measure.",
        payload={
            "G": {"W": Wg.tolist(), "b": bg.tolist()},
            "D": {"w": wd.tolist(), "b": float(bd)},
            "value_history": vals,
            "d_loss": dls,
            "g_loss": gls,
            "samples": fake.tolist(),
            "real_scores": d_real.tolist(),
            "fake_scores": d_fake.tolist(),
            "equilibrium_value": float(2 * np.log(0.5)),
            "mean_gap": gap,
            "initial_mean_gap": initial_gap,
            "g_grad_norm": g_grad_norm,
            "non_saturating": bool(non_saturating),
            "estimate": vals[-1],
            "n": int(m),
            "method": "linear GAN trained by alternating exact gradient steps on the minimax value function",
        },
    )


def cheatsheet():
    return "hmgan: Generative adversarial network: generator vs discriminator minimax"


# compact alias per ledger/NAMING.md
gerongan = geron_gan
