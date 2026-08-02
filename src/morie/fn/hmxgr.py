# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exploding gradients: gradients grow through layers."""

from . import _array_core as np

from ._richresult import RichResult
from .hmvgr import layer_norms

__all__ = ["geron_exploding_gradients"]


def geron_exploding_gradients(grads, tol=2.0, clip_norm=None):
    """
    Exploding gradients: gradients grow through layers.

    Formula: ||grad L^(l)|| -> inf as l decreases when |phi'| > 1 repeatedly

    The mirror image of :func:`morie.fn.hmvgr.geron_vanishing_gradients`,
    and it shares that module's norm computation
    (:func:`morie.fn.hmvgr.layer_norms`) rather than repeating it.
    `grads` is ordered **input side first**, so an exploding network has
    norms that *grow* towards the input. The verdict is the geometric
    mean amplification per layer, which is what compounds with depth.

    When `clip_norm` is given, global norm clipping is applied as well --
    ``g <- g * clip_norm / ||g||_global`` when the global norm exceeds the
    threshold. That rescaling preserves the direction of the update
    exactly, which is why it is preferred to clipping each layer
    separately.

    Parameters
    ----------
    grads : sequence of array-like
        At least 2 layer gradients, input side first.
    tol : float, default 2.0
        Geometric-mean amplification above which the gradient is called
        exploding (> 1).
    clip_norm : float, optional
        Global-norm clipping threshold (> 0).

    Returns
    -------
    result : RichResult
        Keys: norms, ratios, geometric_ratio, amplification, exploding,
        global_norm, clipped, scale, estimate, n, method.

    Examples
    --------
    Norms growing 10x per layer towards the input:

    >>> r = geron_exploding_gradients([[1000.0], [100.0], [10.0], [1.0]])
    >>> [round(float(v), 6) for v in r["ratios"]]
    [0.1, 0.1, 0.1]
    >>> round(float(r["geometric_ratio"]), 9)
    10.0
    >>> bool(r["exploding"])
    True

    Global-norm clipping rescales every layer by the same factor, so the
    clipped global norm is exactly the threshold:

    >>> r2 = geron_exploding_gradients([[3.0], [4.0]], clip_norm=1.0)
    >>> round(float(r2["global_norm"]), 12)
    5.0
    >>> round(float(r2["scale"]), 12)
    0.2
    >>> [round(float(g[0]), 12) for g in r2["clipped"]]
    [0.6, 0.8]

    References
    ----------
    Géron Ch 11
    """
    layers = list(grads)
    if len(layers) < 2:
        raise ValueError("geron_exploding_gradients: need at least 2 layer gradients to measure growth")
    try:
        norms = layer_norms(layers)
    except ValueError as exc:
        raise ValueError(f"geron_exploding_gradients: {exc}") from None
    if np.any(norms == 0):
        zero = np.flatnonzero(norms == 0).tolist()
        raise ValueError(
            f"geron_exploding_gradients: layer(s) {zero} have exactly zero gradient; the ratio is undefined"
        )
    t = float(tol)
    if not (t > 1.0):
        raise ValueError(f"geron_exploding_gradients: tol must be > 1, got {t}")

    ratios = norms[1:] / norms[:-1]
    geo = float(np.exp(np.mean(np.log(norms[:-1] / norms[1:]))))
    amp = float(norms[0] / norms[-1])
    global_norm = float(np.sqrt(np.sum(norms * norms)))

    clipped = None
    scale = 1.0
    if clip_norm is not None:
        c = float(clip_norm)
        if not np.isfinite(c) or c <= 0:
            raise ValueError(f"geron_exploding_gradients: clip_norm must be positive and finite, got {c}")
        scale = min(1.0, c / global_norm)
        clipped = [np.asarray(g, dtype=float) * scale for g in layers]

    return RichResult(
        title="Exploding-gradient diagnosis",
        summary_lines=[
            ("Layers", len(norms)),
            ("Geometric amplification per layer (towards input)", geo),
            ("Global norm", global_norm),
            ("Verdict", "exploding" if geo > t else "healthy"),
        ],
        interpretation=(
            "Exploding gradients announce themselves (NaN losses, wild jumps) where vanishing ones fail "
            "silently; global-norm clipping fixes the magnitude while keeping the update's direction."
        ),
        payload={
            "norms": norms,
            "ratios": ratios,
            "geometric_ratio": geo,
            "amplification": amp,
            "exploding": bool(geo > t),
            "global_norm": global_norm,
            "clipped": clipped,
            "scale": float(scale),
            "tol": t,
            "estimate": geo,
            "n": int(len(norms)),
            "method": "Per-layer gradient norms with geometric amplification and optional global-norm clipping",
        },
    )


def cheatsheet():
    return "hmxgr: Exploding gradients: gradients grow through layers"
