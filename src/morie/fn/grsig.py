# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logistic/sigmoid activation used as link in binary logistic regression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_sigmoid"]

_METHOD = "Logistic (sigmoid) activation"


def geron_sigmoid(t):
    r"""Logistic function, computed in the numerically stable branch form.

    .. math::
        \sigma(t) = \frac{1}{1 + e^{-t}}

    Written naively this overflows for ``t`` around -750.  The two-branch
    form -- :math:`1/(1+e^{-t})` for :math:`t \ge 0` and
    :math:`e^{t}/(1+e^{t})` for :math:`t < 0` -- only ever exponentiates a
    non-positive number, so it saturates to 0 or 1 instead of raising.

    Parameters
    ----------
    t : array-like
        Logit(s), any shape. Must be finite.

    Returns
    -------
    RichResult
        Payload keys ``sigma``, ``derivative`` (:math:`\sigma(1-\sigma)`),
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-14 (Logistic function).

    Examples
    --------
    >>> r = geron_sigmoid([0.0, 2.0, -2.0])
    >>> [round(v, 7) for v in r["sigma"]]
    [0.5, 0.8807971, 0.1192029]
    >>> round(r["derivative"][0], 7)
    0.25

    No overflow at the far tail:

    >>> geron_sigmoid(-800.0)["estimate"]
    0.0
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        raise ValueError("t is empty; sigmoid needs at least one logit.")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains non-finite values.")

    pos = t >= 0
    out = np.empty_like(t, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    e = np.exp(t[~pos])
    out[~pos] = e / (1.0 + e)
    deriv = out * (1.0 - out)

    est = float(out) if out.ndim == 0 else out.tolist()
    return RichResult(
        title="Logistic function",
        summary_lines=[("n", int(t.size)), ("mean sigma", float(out.mean()))],
        payload={
            "sigma": est,
            "derivative": float(deriv) if deriv.ndim == 0 else deriv.tolist(),
            "estimate": est,
            "n": int(t.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsig: sigma(t) = 1/(1+exp(-t)), stable two-branch form; derivative sigma(1-sigma)"
