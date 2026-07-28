# morie.fn -- function file (rootcoder007/morie)
"""Quantitative estimate of drug-likeness."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["qed_score"]

# Bickerton et al. (2012), Table 1: ADS parameters for the eight
# descriptors, in the order MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS.
_ADS = {
    "MW":     (2.817065973, 392.5754953, 290.7489764, 2.419764353,
               49.22325677, 65.37051707, 104.9805561),
    "ALOGP":  (3.172690585, 137.8624751, 2.534937431, 4.581497897,
               0.822739154, 0.576295591, 131.3186604),
    "HBA":    (2.948620388, 160.4605972, 3.615294657, 4.435986202,
               0.290141953, 1.300669958, 148.7763046),
    "HBD":    (1.618662227, 1010.051101, 0.985094388, 0.000000001,
               0.713820843, 0.920922555, 258.1632616),
    "PSA":    (1.876861559, 125.2232657, 62.90773554, 87.83366614,
               12.01999824, 28.51324732, 104.5686167),
    "ROTB":   (0.010000000, 272.4121427, 2.558379970, 1.565547684,
               1.271567166, 2.758063707, 105.4420403),
    "AROM":   (3.217788970, 957.7374108, 2.274627939, 0.000000001,
               1.317690384, 0.375760881, 312.3372610),
    "ALERTS": (0.010000000, 1199.094025, -0.09002883, 0.000000001,
               0.185904477, 0.875193782, 417.7253140),
}
# Weights for QED(w), the "weighted" variant of the paper
_W = {"MW": 0.66, "ALOGP": 0.46, "HBA": 0.05, "HBD": 0.61,
      "PSA": 0.06, "ROTB": 0.65, "AROM": 0.48, "ALERTS": 0.95}
_ORDER = ("MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS")


def _ads(x, p):
    a, b, c, d, e, f, dx_max = p
    try:
        left = a + b / (1.0 + math.exp(-(x - c + d / 2.0) / e))
        right = 1.0 - 1.0 / (1.0 + math.exp(-(x - c - d / 2.0) / f))
        return (left * right) / dx_max
    except OverflowError:
        return 1e-9


def qed_score(properties, weights=None, unweighted=False):
    r"""Desirability-weighted drug-likeness on the Bickerton scale.

    Each of eight physicochemical descriptors is mapped through an
    asymmetric double sigmoid to a desirability :math:`d_i \in (0,1]`,
    and the score is the weighted geometric mean

    .. math::
       QED = \exp\left(\frac{\sum_i w_i \ln d_i}{\sum_i w_i}\right).

    The GEOMETRIC mean is the design decision that matters. An
    arithmetic mean lets a molecule compensate for one disqualifying
    property with several good ones; the geometric mean cannot, because
    a single desirability near zero drags the whole score down. That
    matches how medicinal chemists actually reject compounds, and it is
    why QED behaves differently from a Lipinski rule count.

    ``limiting_descriptor`` names the property holding the score back,
    which is the actionable output -- a QED of 0.3 says little, while
    "0.3, limited by ALERTS" says what to change.

    QED is a measure of RESEMBLANCE to oral drugs, not of activity or
    safety. A high score on an inactive molecule means nothing, and
    plenty of approved drugs score low; injectables and biologics sit
    outside the training distribution entirely.

    Parameters
    ----------
    properties : mapping
        Keys ``MW``, ``ALOGP``, ``HBA``, ``HBD``, ``PSA``, ``ROTB``,
        ``AROM``, ``ALERTS``.
    weights : mapping, optional
        Overrides the published weights.
    unweighted : bool
        Use equal weights, the paper's QED(u).

    Returns
    -------
    RichResult
        ``qed``, ``desirabilities``, ``limiting_descriptor``,
        ``weights``, ``variant``.

    References
    ----------
    Bickerton, Paolini, Besnard, Muresan and Hopkins (2012),
    "Quantifying the chemical beauty of drugs", *Nature Chemistry*
    4:90-98, Table 1 for the ADS parameters.

    Examples
    --------
    >>> p = dict(MW=300, ALOGP=2.5, HBA=4, HBD=1, PSA=60, ROTB=4,
    ...          AROM=2, ALERTS=0)
    >>> bool(0.0 < qed_score(p)["qed"] <= 1.0)
    True
    """
    if not hasattr(properties, "get"):
        raise TypeError("properties must be a mapping of descriptor names.")
    missing = [k for k in _ORDER if properties.get(k) is None]
    if missing:
        raise ValueError(
            "missing descriptor(s): %s. QED needs all eight."
            % ", ".join(missing)
        )
    if unweighted:
        w = {k: 1.0 for k in _ORDER}
    elif weights is None:
        w = dict(_W)
    else:
        w = {k: float(weights.get(k, _W[k])) for k in _ORDER}
    if any(v < 0 for v in w.values()):
        raise ValueError("weights must be non-negative.")
    tot = sum(w[k] for k in _ORDER)
    if tot <= 0:
        raise ValueError("weights must not all be zero.")

    d = {}
    for k in _ORDER:
        val = float(properties[k])
        d[k] = max(min(_ads(val, _ADS[k]), 1.0), 1e-9)

    acc = sum(w[k] * math.log(d[k]) for k in _ORDER) / tot
    qed = math.exp(acc)
    limiting = min(_ORDER, key=lambda k: d[k] if w[k] > 0 else 2.0)
    return RichResult(
        payload={
            "estimate": float(qed),
            "qed": float(qed),
            "desirabilities": d,
            "limiting_descriptor": limiting,
            "limiting_value": float(d[limiting]),
            "limiting_note": (
                "the descriptor holding the score back; a QED alone says "
                "little, while naming the limit says what to change"
            ),
            "weights": w,
            "variant": ("QED(u), unweighted" if unweighted
                        else "QED(w), published weights"),
            "geometric_note": (
                "a weighted GEOMETRIC mean, so one near-zero desirability "
                "drags the whole score down and cannot be compensated for; "
                "an arithmetic mean would let it be"
            ),
            "interpretation_note": (
                "QED measures resemblance to oral small-molecule drugs, not "
                "activity or safety; approved injectables and biologics fall "
                "outside the distribution it was fitted on"
            ),
            "method": "Quantitative estimate of drug-likeness",
        }
    )


def cheatsheet():
    return (
        "qedscr: Bickerton QED as a weighted geometric mean of eight "
        "desirabilities, naming the limiting one"
    )
