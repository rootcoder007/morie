# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dropout: randomly zero units during training with probability p."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dropout"]


def geron_dropout(x, p, training=True, seed=0):
    """
    Dropout: randomly zero units during training with probability p.

    Formula: y = mask * x / (1 - p); mask_i ~ Bernoulli(1-p)

    This is *inverted* dropout: the surviving units are scaled up by
    ``1/(1-p)`` at training time, so ``E[y] = x`` and inference needs no
    rescaling at all -- which is why ``training=False`` returns ``x``
    untouched. The mask comes from a deterministic LCG seeded by ``seed``,
    so a call is reproducible.

    Parameters
    ----------
    x : array-like
        Activations.
    p : float
        Drop probability in [0, 1).
    training : bool, default True
        If False the input passes through unchanged.
    seed : int, default 0
        Seed for the deterministic mask.

    Returns
    -------
    result : RichResult
        Keys: y, mask, scale, n_dropped, drop_fraction, expectation_ok,
        estimate, n, method.

    Examples
    --------
    ``p = 0`` keeps everything and scales by 1:

    >>> r = geron_dropout([1.0, 2.0, 3.0], p=0.0)
    >>> [float(v) for v in r["y"]]
    [1.0, 2.0, 3.0]
    >>> r["n_dropped"]
    0

    At inference the input is returned as is, whatever ``p``:

    >>> [float(v) for v in geron_dropout([1.0, 2.0], p=0.9, training=False)["y"]]
    [1.0, 2.0]

    A surviving unit at ``p = 0.5`` is doubled, and a dropped one is zero:

    >>> r2 = geron_dropout([4.0, 4.0, 4.0, 4.0], p=0.5, seed=7)
    >>> float(r2["scale"])
    2.0
    >>> sorted(set(float(v) for v in r2["y"])) in ([0.0, 8.0], [0.0], [8.0])
    True

    References
    ----------
    Géron Ch 11
    """
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.size == 0:
        raise ValueError("geron_dropout: x is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_dropout: x contains non-finite values")
    pr = float(p)
    if not (0.0 <= pr < 1.0):
        raise ValueError(f"geron_dropout: p must lie in [0, 1), got {p!r} (p = 1 would zero the whole layer)")

    if not training:
        return RichResult(
            title="Dropout (inference)",
            summary_lines=[("p", pr), ("Mode", "inference")],
            payload={
                "y": a.copy(),
                "mask": np.ones_like(a),
                "scale": 1.0,
                "n_dropped": 0,
                "drop_fraction": 0.0,
                "p": pr,
                "training": False,
                "expectation_ok": True,
                "estimate": float(np.mean(a)),
                "n": int(a.size),
                "method": "inverted dropout, inference pass-through",
            },
        )

    st = int(seed) % 2**32
    u = np.empty(a.size)
    for i in range(a.size):
        st = (1664525 * st + 1013904223) % 2**32
        u[i] = (st + 0.5) / 2**32
    mask = (u.reshape(a.shape) >= pr).astype(float)
    scale = 1.0 / (1.0 - pr)
    y = mask * a * scale
    dropped = int(a.size - mask.sum())

    return RichResult(
        title="Dropout (training)",
        summary_lines=[("p", pr), ("Dropped", dropped), ("Scale", scale)],
        interpretation="Inverted dropout keeps E[y] = x, so no rescaling is needed at inference.",
        payload={
            "y": y,
            "mask": mask,
            "scale": float(scale),
            "n_dropped": dropped,
            "drop_fraction": float(dropped / a.size),
            "p": pr,
            "training": True,
            "expectation_ok": True,
            "estimate": float(np.mean(y)),
            "n": int(a.size),
            "method": "inverted dropout y = mask * x / (1 - p)",
        },
    )


def cheatsheet():
    return "hmdrp: Dropout: randomly zero units during training with probability p"
