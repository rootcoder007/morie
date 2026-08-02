# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vanishing gradients: small gradients shrink through many layers."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_vanishing_gradients", "layer_norms"]


def layer_norms(grads):
    """Per-layer gradient L2 norms, ordered input side first."""
    out = []
    for i, g in enumerate(grads):
        a = np.asarray(g, dtype=float)
        if a.size == 0:
            raise ValueError(f"layer {i} has an empty gradient")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"layer {i} gradient contains non-finite values")
        out.append(float(np.sqrt(np.sum(a * a))))
    return np.asarray(out, dtype=float)


def geron_vanishing_gradients(grads, tol=0.5):
    """
    Vanishing gradients: small gradients shrink through many layers.

    Formula: ||grad L^(l)|| -> 0 as l decreases when |phi'| < 1 repeatedly

    Measures the decay instead of asserting it. `grads` is the list of
    per-layer gradients ordered **input side first**, so a vanishing
    network has norms that grow with the index. The per-layer ratio
    ``||g_{l+1}|| / ||g_l||`` is reported, together with the geometric
    mean ratio -- the constant factor per layer -- which is the number
    that actually determines whether a 50-layer stack trains: a factor of
    0.5 per layer is a 2^-50 attenuation.

    The verdict uses the geometric mean, not any single layer, because
    one flat layer proves nothing.

    Parameters
    ----------
    grads : sequence of array-like
        At least 2 layer gradients, input side first.
    tol : float, default 0.5
        Geometric-mean ratio below which the gradient is called vanishing
        (must lie in (0, 1)).

    Returns
    -------
    result : RichResult
        Keys: norms, ratios, geometric_ratio, attenuation, vanishing,
        estimate, n, method.

    Examples
    --------
    Norms shrinking by exactly 100x per layer towards the input:

    >>> r = geron_vanishing_gradients([[1e-6], [1e-4], [1e-2], [1.0]])
    >>> [round(float(v), 6) for v in r["ratios"]]
    [100.0, 100.0, 100.0]
    >>> round(float(r["geometric_ratio"]), 9)
    0.01
    >>> bool(r["vanishing"])
    True
    >>> round(float(r["attenuation"]), 9)
    1e-06

    A healthy stack keeps its scale and is not flagged:

    >>> bool(geron_vanishing_gradients([[1.0], [1.1], [0.9], [1.0]])["vanishing"])
    False

    References
    ----------
    Géron Ch 11
    """
    layers = list(grads)
    if len(layers) < 2:
        raise ValueError("geron_vanishing_gradients: need at least 2 layer gradients to measure a decay")
    try:
        norms = layer_norms(layers)
    except ValueError as exc:
        raise ValueError(f"geron_vanishing_gradients: {exc}") from None
    if np.any(norms == 0):
        zero = np.flatnonzero(norms == 0).tolist()
        raise ValueError(
            f"geron_vanishing_gradients: layer(s) {zero} have exactly zero gradient; the ratio is undefined "
            "(that is a dead layer, not a vanishing one)"
        )
    t = float(tol)
    if not (0.0 < t < 1.0):
        raise ValueError(f"geron_vanishing_gradients: tol must lie in (0, 1), got {t}")

    ratios = norms[1:] / norms[:-1]
    geo = float(np.exp(np.mean(np.log(norms[:-1] / norms[1:]))))
    atten = float(norms[0] / norms[-1])

    return RichResult(
        title="Vanishing-gradient diagnosis",
        summary_lines=[
            ("Layers", len(norms)),
            ("Geometric ratio per layer (towards input)", geo),
            ("Total attenuation input/output", atten),
            ("Verdict", "vanishing" if geo < t else "healthy"),
        ],
        interpretation=(
            "Saturating activations have |phi'| < 1, and the chain rule multiplies those factors once "
            "per layer -- which is why depth, not width, is what kills the early layers."
        ),
        payload={
            "norms": norms,
            "ratios": ratios,
            "geometric_ratio": geo,
            "attenuation": atten,
            "vanishing": bool(geo < t),
            "tol": t,
            "estimate": geo,
            "n": int(len(norms)),
            "method": "Per-layer gradient norms with the geometric mean decay factor towards the input",
        },
    )


def cheatsheet():
    return "hmvgr: Vanishing gradients: small gradients shrink through many layers"
