# morie.fn -- function file (rootcoder007/morie)
r"""TSB: intermittent demand without the obsolescence blind spot.

Intermittent demand is two processes at once -- whether a demand occurs
and how big it is -- so Croston's method splits them and smooths each
separately. The trouble is *when* it smooths.

**Croston updates only when demand occurs, and that is the flaw.**
It tracks the demand size :math:`z` and the inter-demand *interval*
:math:`x`, both updated at demand epochs, and forecasts
:math:`\hat Y = z'/x'`. An item that stops selling entirely therefore
keeps its last forecast **forever**: nothing updates, because nothing
happens. For obsolescence -- the case that matters most in inventory --
the method is silent by construction.

**TSB replaces the interval with the probability, and updates every
period.** Writing :math:`p_t = 1\{Y_t > 0\}`,

.. math::
   p'_t &= p'_{t-1} + \beta\,(p_t - p'_{t-1}) \quad\text{every period},\\
   z'_t &= z'_{t-1} + \alpha\,(z_t - z'_{t-1})
   \quad\text{only when } Y_t > 0,\\
   \hat Y_t &= p'_t\, z'_t.

The probability can be updated on a zero; an interval cannot. So a dying
item decays toward zero at rate :math:`(1-\beta)` per period, which the
anchor measures directly against the closed form.

**And the product form is unbiased where the ratio is not.** Because
:math:`p'` and :math:`z'` are independent under stationary demand,
:math:`E[\hat Y] = E[p']E[z'] = p\mu` exactly. Croston's ratio suffers
an inversion bias, :math:`1/E[X] \ne E[1/X]`, which over-forecasts; SBA
patches it with a deflator :math:`(1-\alpha/2)` that is linear in the
smoothing constant and leaves some bias behind. All three are here, and
the anchor measures the bias of each against a known :math:`p\mu`
rather than repeating the claim.

References
----------
Teunter, R. H., Syntetos, A. A. & Babai, M. Z. (2011) "Intermittent
demand: Linking forecasting to inventory obsolescence", *European
Journal of Operational Research* 214(3), 606-615,
doi:10.1016/j.ejor.2011.05.018. Secs. 2-3: the method, its
unbiasedness, and the obsolescence argument.

Croston, J. D. (1972) "Forecasting and Stock Control for Intermittent
Demands", *Operational Research Quarterly* 23(3), 289-303,
doi:10.2307/3007885. The method TSB modifies.

Syntetos, A. A. & Boylan, J. E. (2005) "The accuracy of intermittent
demand estimates", *International Journal of Forecasting* 21(2),
303-314, doi:10.1016/j.ijforecast.2004.10.001. The SBA deflator.

Syntetos, A. A. & Boylan, J. E. (2001) "On the bias of intermittent
demand estimates", *International Journal of Production Economics*
71(1-3), 457-466, doi:10.1016/S0925-5273(00)00143-2. The inversion
bias itself.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tsb_forecast", "croston_forecast", "sba_forecast",
           "intermittent_forecast", "demand_classification"]

_EPS = 1e-12
_METHODS = ("tsb", "croston", "sba")


def _init(y):
    """Initialise from the first positive demand, as the methods do."""
    pos = [v for v in y if v > 0.0]
    if not pos:
        raise ValueError("tsbF: the series has no positive demand")
    first = next(i for i, v in enumerate(y) if v > 0.0)
    gaps = []
    last = first
    for i in range(first + 1, len(y)):
        if y[i] > 0.0:
            gaps.append(i - last)
            last = i
    z0 = pos[0]
    x0 = (sum(gaps) / len(gaps)) if gaps else 1.0
    p0 = len(pos) / float(len(y))
    return first, z0, max(x0, 1.0), p0


def tsb_forecast(y, alpha=0.1, beta=0.05, horizon=1):
    r"""TSB: probability updated EVERY period, size only on demand."""
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    if n < 2:
        raise ValueError("tsbF: need at least 2 observations, got %d"
                         % n)
    for nm, v in (("alpha", alpha), ("beta", beta)):
        if not 0.0 < float(v) <= 1.0:
            raise ValueError("tsbF: %s must be in (0, 1], got %r"
                             % (nm, v))
    first, z0, _, p0 = _init(yv)
    a, b = float(alpha), float(beta)
    z, p = z0, p0
    fitted, probs, sizes = [], [], []
    for t in range(n):
        occ = 1.0 if yv[t] > 0.0 else 0.0
        # the probability updates on EVERY period, including zeros --
        # this is what lets a dying item decay
        p = p + b * (occ - p)
        if occ:
            z = z + a * (yv[t] - z)
        probs.append(p)
        sizes.append(z)
        fitted.append(p * z)
    return RichResult(payload={
        "estimate": [fitted[-1]] * int(horizon),
        "forecast": [fitted[-1]] * int(horizon),
        "fitted": fitted, "probability": probs, "size": sizes,
        "p_final": p, "z_final": z, "alpha": a, "beta": b,
        "method": "TSB, Teunter, Syntetos & Babai (2011)",
        "updates_on_zeros": True,
    })


def croston_forecast(y, alpha=0.1, horizon=1):
    r"""Croston: size and INTERVAL, updated only at demand epochs.

    Nothing updates on a zero, so an obsolete item keeps its last
    forecast indefinitely -- which is the failure TSB exists to fix.
    """
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    if n < 2:
        raise ValueError("tsbF: need at least 2 observations, got %d"
                         % n)
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("tsbF: alpha must be in (0, 1], got %r"
                         % (alpha,))
    first, z0, x0, _ = _init(yv)
    a = float(alpha)
    z, x = z0, x0
    since = 0
    fitted = []
    for t in range(n):
        since += 1
        if yv[t] > 0.0:
            z = z + a * (yv[t] - z)
            x = x + a * (since - x)
            since = 0
        fitted.append(z / max(x, _EPS))
    return RichResult(payload={
        "estimate": [fitted[-1]] * int(horizon),
        "forecast": [fitted[-1]] * int(horizon),
        "fitted": fitted, "z_final": z, "x_final": x, "alpha": a,
        "method": "Croston (1972)", "updates_on_zeros": False,
    })


def sba_forecast(y, alpha=0.1, horizon=1):
    r"""Syntetos-Boylan: Croston deflated by :math:`(1-\alpha/2)`."""
    c = croston_forecast(y, alpha=alpha, horizon=horizon)
    d = 1.0 - float(alpha) / 2.0
    return RichResult(payload={
        "estimate": [v * d for v in c["forecast"]],
        "forecast": [v * d for v in c["forecast"]],
        "fitted": [v * d for v in c["fitted"]],
        "deflator": d, "alpha": float(alpha),
        "method": "Syntetos-Boylan Approximation (2005)",
        "updates_on_zeros": False,
    })


def demand_classification(y, adi_cut=1.32, cv2_cut=0.49):
    r"""The Syntetos-Boylan-Croston categories.

    ADI is the average inter-demand interval and CV^2 the squared
    coefficient of variation of the non-zero sizes; the cuts are the
    published ones.
    """
    yv = [float(v) for v in k.vec(y)]
    pos = [v for v in yv if v > 0.0]
    if len(pos) < 2:
        raise ValueError("tsbF: need at least 2 positive demands")
    adi = len(yv) / float(len(pos))
    mu = sum(pos) / len(pos)
    cv2 = (k.sd(pos) / mu) ** 2 if mu > 0 else 0.0
    if adi <= adi_cut and cv2 <= cv2_cut:
        cls = "smooth"
    elif adi <= adi_cut:
        cls = "erratic"
    elif cv2 <= cv2_cut:
        cls = "intermittent"
    else:
        cls = "lumpy"
    return {"class": cls, "adi": adi, "cv2": cv2,
            "n_positive": len(pos), "n": len(yv)}


def intermittent_forecast(y, method="tsb", alpha=0.1, beta=0.05,
                          horizon=1):
    """Dispatch, so the three can be compared on one series."""
    if method not in _METHODS:
        raise ValueError("tsbF: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    if method == "tsb":
        return tsb_forecast(y, alpha=alpha, beta=beta, horizon=horizon)
    if method == "croston":
        return croston_forecast(y, alpha=alpha, horizon=horizon)
    return sba_forecast(y, alpha=alpha, horizon=horizon)


def cheatsheet():
    return ("tsbF: TSB updates the PROBABILITY every period (p' += "
            "beta(occ - p')) and the SIZE only on demand; forecast is "
            "the PRODUCT p'z', which is unbiased because the two are "
            "independent. Croston smooths the INTERVAL and forecasts "
            "z'/x' -- nothing updates on a zero, so an obsolete item "
            "keeps its forecast forever, and the ratio carries an "
            "inversion bias. SBA deflates by (1 - alpha/2).")


# compact alias per ledger/NAMING.md
tsbforecast = tsb_forecast
