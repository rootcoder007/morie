# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GAN minimax objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gan_minimax"]

_METHOD = "GAN minimax objective and per-player losses"


def geron_gan_minimax(real, fake, D_real, D_fake, eps=1e-12):
    r"""The two-player value function and each player's loss.

    .. math::
        \min_G \max_D\ \mathbb E_x[\log D(x)]
        + \mathbb E_z[\log(1 - D(G(z)))]

    The discriminator climbs ``V``; the generator was originally meant
    to descend it.  In practice it does not: early in training
    ``D(G(z))`` is near 0, where ``log(1 - D(G(z)))`` is flat, so the
    generator gets almost no gradient exactly when it needs the most.
    The non-saturating substitute -- maximise ``log D(G(z))`` -- has the
    same fixed point and a usable gradient, and is what everyone
    actually trains with.  Both are reported.

    At the optimum ``D = 0.5`` everywhere and ``V = -2 log 2 =
    -1.386``, the reference point for "the discriminator has stopped
    learning anything".

    Parameters
    ----------
    real, fake : array-like
        The samples themselves; used only for bookkeeping (counts), so
        the objective depends solely on the discriminator outputs.
    D_real, D_fake : array-like
        Discriminator probabilities in ``[0, 1]``.
    eps : float, optional
        Clip guarding ``log 0``, default ``1e-12``.

    Returns
    -------
    RichResult
        Payload keys ``value``, ``d_loss``, ``g_loss_nonsaturating``,
        ``g_loss_saturating``, ``d_accuracy``, ``at_equilibrium``
        (``V`` within 1e-6 of ``-2 log 2``), ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 18, GAN section (Goodfellow et al. 2014).

    Examples
    --------
    A discriminator reduced to coin-flipping is the theoretical
    equilibrium:

    >>> r = geron_gan_minimax([1.0], [0.0], [0.5], [0.5])
    >>> round(r["value"], 10)
    -1.3862943611
    >>> r["at_equilibrium"]
    True
    >>> round(r["g_loss_nonsaturating"], 10)
    0.6931471806

    A discriminator that is winning gives the generator a large
    non-saturating loss but a nearly flat saturating one -- the vanishing
    gradient the substitute exists to avoid:

    >>> r2 = geron_gan_minimax([1.0], [0.0], [0.99], [0.01])
    >>> round(r2["g_loss_nonsaturating"], 6)
    4.60517
    >>> round(r2["g_loss_saturating"], 6)
    -0.01005
    >>> r2["d_accuracy"]
    1.0
    """
    dr = np.atleast_1d(np.asarray(D_real, dtype=float)).ravel()
    df = np.atleast_1d(np.asarray(D_fake, dtype=float)).ravel()
    if dr.size == 0 or df.size == 0:
        raise ValueError("D_real and D_fake must be non-empty.")
    for name, arr in (("D_real", dr), ("D_fake", df)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite.")
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} holds probabilities and must lie in [0, 1].")
    eps = float(eps)
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps must lie in (0, 0.5), got {eps}.")

    drc = np.clip(dr, eps, 1.0 - eps)
    dfc = np.clip(df, eps, 1.0 - eps)
    value = float(np.mean(np.log(drc)) + np.mean(np.log(1.0 - dfc)))
    d_loss = -value
    g_nonsat = float(-np.mean(np.log(dfc)))
    g_sat = float(np.mean(np.log(1.0 - dfc)))
    acc = float((np.mean(dr >= 0.5) + np.mean(df < 0.5)) / 2.0)

    return RichResult(
        title="GAN minimax objective",
        summary_lines=[("V(D, G)", value), ("D loss", d_loss),
                       ("G loss (non-sat)", g_nonsat)],
        payload={
            "value": value,
            "d_loss": d_loss,
            "g_loss_nonsaturating": g_nonsat,
            "g_loss_saturating": g_sat,
            "d_accuracy": acc,
            "at_equilibrium": bool(abs(value + 2.0 * np.log(2.0)) < 1e-6),
            "n_real": int(np.asarray(real).shape[0]) if np.asarray(real).ndim else 1,
            "n_fake": int(np.asarray(fake).shape[0]) if np.asarray(fake).ndim else 1,
            "estimate": value,
            "n": int(dr.size + df.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgan: V = E[log D(x)] + E[log(1-D(G(z)))]; non-saturating G loss = -E[log D(G(z))]"
