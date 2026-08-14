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
bias itself -- an ASYMPTOTIC result, which is why the anchor measures
it under ``init="known"``.

Prak, D., Teunter, R., Babai, M. Z., Boylan, J. E. & Syntetos, A.
(2021) "Robust compound Poisson parameter estimation for inventory
control", *Omega* 104, 102481, doi:10.1016/j.omega.2021.102481. That
the standard intermittent-demand estimators are severely biased in
finite samples, which is the effect ``burn_in`` and the ``init``
routes exist to separate from the asymptotic bias above.

Teunter, R. H. & Duncan, L. (2009) "Forecasting intermittent demand:
a comparative study", *Journal of the Operational Research Society*
60(3), 321-329, doi:10.1057/palgrave.jors.2602569. That per-period
error measures are the wrong yardstick for intermittent demand, which
is why the anchor compares bias against a known p*mu rather than
ranking methods on RMSE.

Kourentzes, N. (2014) "On intermittent demand model optimisation and
selection", *International Journal of Production Economics* 156,
180-190. The article prints no DOI. That the smoothing constants
and the initial states should be estimated together, and that the
usual squared-error optimisation misbehaves on intermittent series.

Babai, M. Z., Syntetos, A. & Teunter, R. (2014) "Intermittent demand
forecasting: An empirical study on accuracy and the risk of
obsolescence", *International Journal of Production Economics* 157,
212-219. The article prints no DOI. The empirical comparison of
TSB against Croston and SBA under obsolescence risk.

Babai, M. Z., Dallery, Y., Boubaker, S. & Kalai, R. (2019) "A new
method to forecast intermittent demand in the presence of inventory
obsolescence", *International Journal of Production Economics* 209,
30-41, doi:10.1016/j.ijpe.2018.01.026. A later obsolescence-aware
alternative to TSB.

Yang, Y., Ding, C., Lee, S., Yu, L. & Ma, F. (2021) "A modified
Teunter-Syntetos-Babai method for intermittent demand forecasting",
*Journal of Management Science and Engineering* 6(1), 53-63,
doi:10.1016/j.jmse.2021.02.008. A modification of the TSB update.

Svetunkov, I. & Boylan, J. E. (2023) "iETS: State space model for
intermittent demand forecasting", *International Journal of
Production Economics* 265, 109013, doi:10.1016/j.ijpe.2023.109013.
The state-space formulation that puts Croston and TSB in one family.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tsb_forecast", "croston_forecast", "sba_forecast",
           "intermittent_forecast", "demand_classification"]

_EPS = 1e-12
_METHODS = ("tsb", "croston", "sba")


_INITS = ("global", "heuristic", "known")


def _init(y, init="global", z0=None, x0=None, p0=None):
    r"""Starting states for the smoothers -- three published routes.

    Which one is right depends on what is being measured, and the
    difference is not cosmetic: at a small smoothing constant the
    initial state decays as :math:`(1-\alpha)^t`, so it takes roughly
    :math:`3/\alpha` periods to fall below 5% of its starting weight.
    At :math:`\alpha = 0.05` that is 60 periods. A finite sample
    shorter than that is still reading its own initialisation.

    ``init="known"``
        The states are supplied. In a Monte Carlo study the data
        generating process is known, so setting :math:`z_0 = \mu`,
        :math:`x_0 = 1/p`, :math:`p_0 = p` removes the transient
        entirely and whatever bias survives is the *structural* bias of
        the estimator -- which is what the asymptotic theory is about.
    ``init="global"``
        :math:`z_0` is the mean of the non-zero demands and
        :math:`x_0 = T/N_{>0}`, both over the whole sample. This is the
        closest approximation to the truth available without knowing the
        DGP, so it is the default for real data.
    ``init="heuristic"``
        :math:`z_0` is the FIRST positive demand and :math:`x_0` the
        mean observed gap. This is what a naive implementation does, and
        it is kept because it is what the finite-sample bias literature
        is describing: a low first observation drags the whole path down
        for :math:`\sim 3/\alpha` periods, which can flip the sign of the
        measured bias relative to the asymptotic prediction.
    """
    yv = [float(v) for v in y]
    pos = [v for v in yv if v > 0.0]
    if not pos:
        raise ValueError("tsbF: the series has no positive demand")
    if init not in _INITS:
        raise ValueError("tsbF: init must be one of %s, got %r"
                         % (", ".join(_INITS), init))
    first = next(i for i, v in enumerate(yv) if v > 0.0)
    if init == "known":
        if z0 is None or (x0 is None and p0 is None):
            raise ValueError("tsbF: init='known' needs z0 and one of "
                             "x0 / p0")
        Z = float(z0)
        if x0 is None:
            if not 0.0 < float(p0) <= 1.0:
                raise ValueError("tsbF: p0 must be in (0, 1], got %r"
                                 % (p0,))
            X = 1.0 / float(p0)
            P = float(p0)
        else:
            if float(x0) < 1.0:
                raise ValueError("tsbF: x0 must be at least 1, got %r"
                                 % (x0,))
            X = float(x0)
            P = 1.0 / X if p0 is None else float(p0)
        if Z <= 0.0:
            raise ValueError("tsbF: z0 must be positive, got %r" % (z0,))
        return first, Z, X, P
    if init == "global":
        Z = sum(pos) / len(pos)
        X = len(yv) / float(len(pos))
        P = len(pos) / float(len(yv))
        return first, Z, max(X, 1.0), P
    # heuristic
    gaps = []
    last = first
    for i in range(first + 1, len(yv)):
        if yv[i] > 0.0:
            gaps.append(i - last)
            last = i
    X = (sum(gaps) / len(gaps)) if gaps else 1.0
    return first, pos[0], max(X, 1.0), len(pos) / float(len(yv))


def _burn(seq, burn_in):
    """Drop the first ``burn_in`` fitted values.

    The initial state's weight decays as (1-alpha)^t, so the first
    ~3/alpha fitted values are still dominated by whatever the states
    were set to. Discarding them isolates the estimator's own behaviour.
    """
    b = int(burn_in)
    if b < 0:
        raise ValueError("tsbF: burn_in must be non-negative, got %r"
                         % (burn_in,))
    if b >= len(seq):
        raise ValueError("tsbF: burn_in %d discards the whole series "
                         "of length %d" % (b, len(seq)))
    return seq[b:]


def tsb_forecast(y, alpha=0.1, beta=0.05, horizon=1, init="global",
                 z0=None, p0=None, burn_in=0):
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
    first, zi, _, pi = _init(yv, init=init, z0=z0, p0=p0,
                             x0=None if p0 is None else 1.0 / p0)
    a, b = float(alpha), float(beta)
    z, p = zi, pi
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
        "fitted": _burn(fitted, burn_in),
        "fitted_full": fitted,
        "probability": _burn(probs, burn_in),
        "size": _burn(sizes, burn_in),
        "init": init, "burn_in": int(burn_in),
        "z_init": zi, "p_init": pi,
        "p_final": p, "z_final": z, "alpha": a, "beta": b,
        "method": "TSB, Teunter, Syntetos & Babai (2011)",
        "updates_on_zeros": True,
    })


def croston_forecast(y, alpha=0.1, horizon=1, init="global",
                     z0=None, x0=None, burn_in=0):
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
    first, zi, xi, _ = _init(yv, init=init, z0=z0, x0=x0)
    a = float(alpha)
    z, x = zi, xi
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
        "fitted": _burn(fitted, burn_in), "fitted_full": fitted,
        "z_final": z, "x_final": x, "alpha": a,
        "init": init, "burn_in": int(burn_in),
        "z_init": zi, "x_init": xi,
        "method": "Croston (1972)", "updates_on_zeros": False,
    })


def sba_forecast(y, alpha=0.1, horizon=1, init="global",
                 z0=None, x0=None, burn_in=0):
    r"""Syntetos-Boylan: Croston deflated by :math:`(1-\alpha/2)`."""
    c = croston_forecast(y, alpha=alpha, horizon=horizon,
                         init=init, z0=z0, x0=x0, burn_in=burn_in)
    d = 1.0 - float(alpha) / 2.0
    return RichResult(payload={
        "estimate": [v * d for v in c["forecast"]],
        "forecast": [v * d for v in c["forecast"]],
        "fitted": [v * d for v in c["fitted"]],
        "fitted_full": [v * d for v in c["fitted_full"]],
        "init": init, "burn_in": int(burn_in),
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
                          horizon=1, init="global", z0=None,
                          x0=None, p0=None, burn_in=0):
    """Dispatch, so the three can be compared on one series."""
    if method not in _METHODS:
        raise ValueError("tsbF: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    if method == "tsb":
        return tsb_forecast(y, alpha=alpha, beta=beta,
                            horizon=horizon, init=init, z0=z0,
                            p0=p0, burn_in=burn_in)
    if method == "croston":
        return croston_forecast(y, alpha=alpha, horizon=horizon,
                                init=init, z0=z0, x0=x0,
                                burn_in=burn_in)
    return sba_forecast(y, alpha=alpha, horizon=horizon, init=init,
                        z0=z0, x0=x0, burn_in=burn_in)


def cheatsheet():
    return ("tsbF: TSB updates the PROBABILITY every period (p' += "
            "beta(occ - p')) and the SIZE only on demand; forecast is "
            "the PRODUCT p'z', which is unbiased because the two are "
            "independent. Croston smooths the INTERVAL and forecasts "
            "z'/x' -- nothing updates on a zero, so an obsolete item "
            "keeps its forecast forever, and the ratio carries an "
            "inversion bias. SBA deflates by (1 - alpha/2). The "
            "inversion bias is ASYMPTOTIC: use init='known' to see "
            "it, because with init='heuristic' the initial state "
            "decays as (1-alpha)^t and takes ~3/alpha periods to "
            "clear, which can flip the measured sign (Prak et al. "
            "2021). burn_in drops that transient.")


# compact alias per ledger/NAMING.md
tsbforecast = tsb_forecast
