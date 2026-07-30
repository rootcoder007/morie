# morie.fn -- function file (rootcoder007/morie)
"""DP-Adam: DP-SGD's private gradient with Adam's adaptivity."""

from __future__ import annotations

import numpy as np

from ._optim import init_state
from ._richresult import RichResult
from .dpsgd import dp_sgd

__all__ = ["dp_adam"]


def dp_adam(grads, C=1.0, sigma=1.0, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            state=None, seed=None):
    r"""Adam applied to a differentially private gradient.

    The privatised gradient from :func:`~morie.fn.dpsgd.dp_sgd` is fed through
    the usual Adam moments. Privacy is unaffected: the moments are functions of
    already-private gradients, so this is post-processing and costs nothing
    further.

    Adaptivity interacts badly with DP noise in a way worth knowing. Adam
    divides by :math:`\sqrt{\hat v}`, and under heavy noise :math:`\hat v` is
    dominated by the *noise* variance rather than by signal, so the effective
    per-coordinate step is set by how much noise that coordinate received. In
    the high-noise regime DP-Adam therefore degrades toward a normalised
    random walk, which is why plain DP-SGD is often competitive with it in
    practice despite Adam's usual advantage.

    ``signal_to_noise`` is returned per step as a diagnostic: when it drops
    well below 1, the adaptivity is tracking noise.

    Parameters
    ----------
    grads : array-like
        Per-example gradients ``(B, p)``.
    C : float
        Per-example clipping bound.
    sigma : float
        Noise multiplier.
    lr : float
        Step size.
    betas : tuple
        Adam moment decay rates.
    eps : float
        Denominator floor.
    state : dict, optional
        Adam state from the previous step.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``update``, ``state``, ``private_gradient``, ``signal_to_noise``,
        ``clipped_fraction``.

    References
    ----------
    Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I.,
        Talwar, K., & Zhang, L. (2016). Deep learning with differential
        privacy. *Proceedings of the 2016 ACM SIGSAC Conference on
        Computer and Communications Security*, 308-318.

    Examples
    --------
    With no noise DP-Adam reduces to Adam on the clipped mean gradient.

    >>> import numpy as np
    >>> from morie.fn.adamopt import adam
    >>> rng = np.random.default_rng(0)
    >>> G = rng.normal(size=(32, 4)) * 0.05
    >>> r = dp_adam(G, C=10.0, sigma=0.0, lr=0.01, seed=0)
    >>> a = adam(G.mean(axis=0), lr=0.01)["update"]
    >>> bool(np.allclose(r["update"], a, atol=1e-10))
    True

    State threads between steps as for any Adam variant.

    >>> r2 = dp_adam(G, C=10.0, sigma=0.0, lr=0.01, state=r["state"], seed=0)
    >>> int(r2["state"]["t"])
    2

    Signal-to-noise falls as the noise multiplier rises, which is the
    diagnostic that matters for adaptivity.

    >>> lo = dp_adam(G, C=1.0, sigma=0.1, seed=1)["signal_to_noise"]
    >>> hi = dp_adam(G, C=1.0, sigma=10.0, seed=1)["signal_to_noise"]
    >>> bool(hi < lo)
    True
    """
    step = dp_sgd(grads, C=C, sigma=sigma, lr=1.0, seed=seed)
    g = np.asarray(step["private_gradient"], dtype=float)
    b1, b2 = float(betas[0]), float(betas[1])
    st = init_state(state, g.size, keys=("m", "v"))
    t = st["t"]
    st["m"] = b1 * st["m"] + (1 - b1) * g
    st["v"] = b2 * st["v"] + (1 - b2) * g**2
    m_hat = st["m"] / (1 - b1**t)
    v_hat = st["v"] / (1 - b2**t)
    update = -lr * m_hat / (np.sqrt(v_hat) + eps)
    B = int(step["batch_size"])
    noise_sd = float(sigma * C) / max(B, 1)
    snr = float(np.linalg.norm(g) / max(noise_sd * np.sqrt(g.size), 1e-300))
    return RichResult(
        title="DP-Adam step",
        summary_lines=[("step", int(t)), ("noise sd", noise_sd),
                       ("signal/noise", snr)],
        warnings=(["signal-to-noise is below 1, so Adam's second moment is "
                   "tracking DP noise rather than gradient scale; plain DP-SGD "
                   "is often better in this regime"] if snr < 1.0 else []),
        payload={
            "update": update, "state": st, "m": st["m"], "v": st["v"],
            "private_gradient": g, "signal_to_noise": snr,
            "clipped_fraction": step["clipped_fraction"],
            "noise_sd": noise_sd, "t": int(t), "method": "dp_adam",
        },
    )


def cheatsheet():
    return "dpadam: free post-processing, but under heavy noise v_hat tracks NOISE -- watch signal_to_noise"
