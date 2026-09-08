# morie.fn -- function file (rootcoder007/morie)
"""Renyi entropy of order alpha.

Renyi, A. (1961).  On measures of entropy and information.  Proc. 4th
Berkeley Symp. Math. Statist. Prob. 1:547-561.

Source note.  The Berkeley Symposium volume is open access at Project
Euclid but the scan carries no text layer, so the definition could not
be quoted from the original page.  It was taken instead from the
standard mathematical references (nLab, "Renyi entropy") and is
unambiguous:

    H_alpha(p) = (1/(1 - alpha)) log sum_k p_k^alpha,
    alpha > 0, alpha != 1,

with the two removable cases handled by their limits, alpha -> 1
giving Shannon entropy and alpha -> infinity giving the min-entropy
-log max_k p_k.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["renyi_entropy"]


def _norm(y):
    """Accept counts or probabilities; normalize to a distribution."""
    p = [float(v) for v in y]
    if any(v < 0 for v in p):
        raise ValueError("probabilities must be non-negative")
    tot = sum(p)
    if not tot > 0:
        raise ValueError("total mass must be positive")
    return [v / tot for v in p]


def renyi_entropy(y, alpha=2.0, base=2.0):
    """Entropy of order alpha.

    alpha tunes how much the measure cares about rare outcomes.  At
    alpha = 0 it is the log of the support size and ignores the
    probabilities entirely; at alpha = 1 it is Shannon entropy; as
    alpha grows it is dominated by the single most likely outcome, and
    in the limit it is the min-entropy.  H_alpha is non-increasing in
    alpha.

    Parameters
    ----------
    y : array-like of counts or probabilities.
    alpha : float >= 0, or float("inf") for the min-entropy.
    base : log base; 2 gives bits, ``None`` gives nats.

    Returns
    -------
    RichResult with keys estimate, alpha, base, probabilities,
    support, method.
    """
    p = _norm(y)
    a = float(alpha)
    if a < 0:
        raise ValueError("alpha must be non-negative")
    lb = None if base is None else float(base)
    if lb is not None and (lb <= 0 or lb == 1.0):
        raise ValueError("base must be positive and not 1")

    def _log(v):
        return math.log(v) if lb is None else math.log(v) / math.log(lb)

    sup = sum(1 for v in p if v > 0)
    if a == float("inf"):
        h = -_log(max(p))
    elif a == 1.0:
        h = -sum(v * _log(v) for v in p if v > 0)
    elif a == 0.0:
        h = _log(sup)
    else:
        h = _log(sum(v ** a for v in p if v > 0)) / (1.0 - a)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(h), "alpha": a, "base": lb,
        "probabilities": p, "support": sup,
        "method": "Renyi entropy of order alpha (Renyi 1961)",
    }), "renent")


def cheatsheet():
    return "renent: Renyi entropy of order alpha"


# compact alias per ledger/NAMING.md
renyient = renyi_entropy
