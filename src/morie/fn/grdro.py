# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inverted dropout (training-time)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dropout"]

_METHOD = "Inverted dropout (training time)"


def _lcg_uniforms(count, seed):
    """``count`` uniforms from the reference LCG.

    ``s = (1664525 s + 1013904223) mod 2**32``, ``u = (s + 0.5)/2**32``.
    """
    s = int(seed) % 2**32
    out = np.empty(count, dtype=float)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s + 0.5) / 2**32
    return out


def geron_dropout(a, p, seed=0):
    r"""Drop activations at random, then scale the survivors.

    .. math::
        m \sim \mathrm{Bernoulli}(1-p)^n,\qquad
        a_{\text{out}} = \frac{m \odot a_{\text{in}}}{1-p}

    The division by ``1 - p`` is the "inverted" part and it is what
    makes test time free: since
    :math:`\mathbb E[a_{\text{out}}] = a_{\text{in}}`, the network can
    be run at inference with dropout simply switched off, no rescaling
    of the weights.  ``expectation_ratio`` reports how well that held on
    this particular draw.

    Masks come from the deterministic LCG above, so a given ``seed``
    reproduces the same pattern.

    Parameters
    ----------
    a : array-like
        Activations.
    p : float
        Drop probability in ``[0, 1)``. ``p = 1`` would drop everything
        and divide by zero.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``output``, ``mask``, ``keep_prob``,
        ``fraction_dropped``, ``scale`` (``1/(1-p)``),
        ``expectation_ratio`` (mean of output over mean of input),
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Dropout section (Srivastava et al. 2014).

    Examples
    --------
    ``p = 0`` is the identity, scale 1, nothing dropped:

    >>> r = geron_dropout([1.0, 2.0, 3.0], p=0.0)
    >>> r["output"]
    [1.0, 2.0, 3.0]
    >>> r["scale"]
    1.0

    At ``p = 0.5`` the survivors are doubled, so a kept unit reads 2.0
    where the input was 1.0:

    >>> r2 = geron_dropout([1.0] * 8, p=0.5, seed=1)
    >>> set(r2["output"]) <= {0.0, 2.0}
    True
    >>> r2["scale"]
    2.0
    >>> r2["fraction_dropped"] == 1 - sum(r2["mask"]) / 8
    True
    """
    A = np.asarray(a, dtype=float)
    if A.size == 0:
        raise ValueError("a is empty.")
    if not np.all(np.isfinite(A)):
        raise ValueError("a must be finite.")
    p = float(p)
    if not (0.0 <= p < 1.0):
        raise ValueError(
            f"p is the drop probability and must lie in [0, 1); p = 1 drops every "
            f"unit and divides by zero. Got {p}."
        )

    keep = 1.0 - p
    u = _lcg_uniforms(A.size, seed).reshape(A.shape)
    mask = (u < keep).astype(float)
    out = mask * A / keep

    mean_in = float(np.mean(A))
    ratio = float(np.mean(out) / mean_in) if mean_in != 0 else None

    return RichResult(
        title="Dropout",
        summary_lines=[("p", p), ("Kept", float(mask.mean())), ("Scale", 1.0 / keep)],
        payload={
            "output": out.tolist(),
            "mask": mask.tolist(),
            "keep_prob": keep,
            "fraction_dropped": float(1.0 - mask.mean()),
            "scale": 1.0 / keep,
            "expectation_ratio": ratio,
            "p": p,
            "seed": int(seed),
            "estimate": out.tolist(),
            "n": int(A.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdro: a *= Bernoulli(1-p)/(1-p) -- inverted dropout, so inference needs no rescaling"
