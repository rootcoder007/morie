# morie.fn -- function file (rootcoder007/morie)
"""DP-GAN: privacy through the discriminator only."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .dpsgd import dp_sgd

__all__ = ["dp_gan"]


def dp_gan(disc_grads, C=1.0, sigma=1.0, lr=0.1, n_disc_steps=1, seed=None):
    r"""One privatised discriminator step for differentially private GAN training.

    Only the **discriminator** touches real data, so only its gradients need
    privatising. The generator sees the data solely through the discriminator,
    and by post-processing its updates -- and every sample it ever produces --
    inherit the discriminator's guarantee at no extra cost.

    That asymmetry is the entire design. Adding noise to the generator as well
    would spend budget for nothing; failing to notice it and privatising both
    is a common and expensive mistake.

    The budget must be counted over **discriminator steps**, and GANs use many
    per generator step. ``n_disc_steps`` is recorded so the accounting is
    visible; compose with :func:`~morie.fn.dprnyi.renyi_dp_composition`.

    The practical difficulty is that clipping plus noise destroys exactly the
    fine gradient signal a discriminator needs, so DP-GAN sample quality
    degrades sharply with the budget -- more so than for classifiers.

    Parameters
    ----------
    disc_grads : array-like
        Per-example discriminator gradients ``(B, p)``.
    C : float
        Per-example clipping bound.
    sigma : float
        Noise multiplier.
    lr : float
        Discriminator learning rate.
    n_disc_steps : int
        Discriminator steps per generator step, for accounting.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``disc_update``, ``private_gradient``, ``generator_is_free``,
        ``steps_to_account``, ``clipped_fraction``.

    References
    ----------
    Xie, L., Lin, K., Wang, S., Wang, F., & Zhou, J. (2018). Differentially
        private generative adversarial network. arXiv:1802.06739.

    Examples
    --------
    The generator costs nothing extra -- it only ever sees privatised
    gradients.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> G = rng.normal(size=(64, 6)) * 0.1
    >>> r = dp_gan(G, C=1.0, sigma=1.0, lr=0.05, n_disc_steps=5, seed=0)
    >>> bool(r["generator_is_free"])
    True

    Accounting is over discriminator steps, and that count is surfaced.

    >>> int(r["steps_to_account"])
    5

    The privatised gradient matches DP-SGD on the same input, since that is
    exactly what it is.

    >>> from morie.fn.dpsgd import dp_sgd
    >>> a = dp_sgd(G, C=1.0, sigma=1.0, lr=1.0, seed=0)["private_gradient"]
    >>> bool(np.allclose(r["private_gradient"], a))
    True
    """
    step = dp_sgd(disc_grads, C=C, sigma=sigma, lr=1.0, seed=seed)
    g = np.asarray(step["private_gradient"], dtype=float)
    return RichResult(
        title="DP-GAN discriminator step",
        summary_lines=[("batch", int(step["batch_size"])),
                       ("disc steps to account", int(n_disc_steps)),
                       ("clipped", step["clipped_fraction"])],
        warnings=["only the discriminator needs privatising; noising the "
                  "generator as well spends budget for nothing",
                  "account over discriminator steps, which outnumber "
                  "generator steps"],
        payload={
            "disc_update": -lr * g, "private_gradient": g,
            "generator_is_free": True,
            "steps_to_account": int(n_disc_steps),
            "clipped_fraction": step["clipped_fraction"],
            "noise_sd": step["noise_sd"], "C": float(C),
            "sigma": float(sigma), "method": "dp_gan",
        },
    )


def cheatsheet():
    return "dpgan: privatise the DISCRIMINATOR only; the generator is post-processing and free"
