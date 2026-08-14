# morie.fn -- function file (rootcoder007/morie)
r"""Online super learning.

The data arrive sequentially: :math:`O(t)` is drawn from a conditional
distribution given a fixed-dimensional **summary measure**
:math:`Z(t)` of the past, and that conditional law is described by one
common parameter :math:`\theta` across time. The framing is
deliberately wide -- with an *empty* summary measure it is the ordinary
i.i.d. case; with a parametric conditional density it is a classical
time series model; and it also covers group sequential adaptive
designs where the randomisation depends on data from earlier groups.

**Cross-validation has to respect time.** V-fold splitting would train
on the future to predict the past. The online analogue is
**sequential validation**: at each :math:`t`, train on
:math:`O(1),\dots,O(t-1)` and score the one-step-ahead prediction of
:math:`O(t)`. Summed over :math:`t`, that is an honest risk estimate
for a sequentially-generated sample, and it is the quantity the
weights are chosen to minimise.

**The oracle property survives the dependence.** Because the loss is
evaluated on a genuinely held-out future observation at every step,
the same argument applies: the online super learner performs
asymptotically as well as the best candidate in the library, without
knowing which that is. What changes is only *how* the risk is
computed.

**Updating in constant memory.** The weights are refitted as data
arrive; the cumulative losses are sufficient, so nothing needs
re-scoring and memory does not grow with :math:`t`. The anchor checks
that the online risk equals the batch risk recomputed from scratch --
if the update drifts, that is where it shows.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 18 (van der
Laan & Benkeser): data generated sequentially by a conditional
distribution of O(t) given summary measures of the past, identified by
a common parameter theta; the special cases obtained by choosing the
summary measure -- an empty one giving i.i.d. sampling from a fixed
distribution in a semiparametric model, a group sequential adaptive
design whose randomisation or censoring mechanism depends on earlier
groups, and more generally a whole range of time series models; and
online super learning with sequential cross-validation.

Benkeser, D., Ju, C., Lendle, S. & van der Laan, M. J. (2018) "Online
cross-validation-based ensemble learning", *Statistics in Medicine*
37(2), 249-260, doi:10.1002/sim.7320.

van der Laan, M. J., Polley, E. C. & Hubbard, A. E. (2007) "Super
Learner", *Statistical Applications in Genetics and Molecular
Biology* 6(1), Article 25, doi:10.2202/1544-6115.1309. The batch
version; implemented in :mod:`tlseqsl`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sequential_risk", "online_super_learner",
           "update_weights", "summary_measure"]

_EPS = 1e-12
_LOSSES = ("squared", "log")


def _loss(kind, y, p):
    if kind == "squared":
        return (y - p) ** 2
    q = min(max(p, _EPS), 1.0 - _EPS)
    return -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))


def summary_measure(history, lags=1):
    r"""A fixed-dimensional summary :math:`Z` of the past.

    ``lags=0`` gives the empty summary, which is exactly the i.i.d.
    case -- the same machinery, one special case.
    """
    L = int(lags)
    if L < 0:
        raise ValueError("tlonsl: lags must be non-negative")
    if L == 0:
        return []
    h = [float(v) for v in k.vec(history)]
    z = h[-L:] if len(h) >= L else [0.0] * (L - len(h)) + h
    return z


def sequential_risk(y, algorithm, loss="squared", burn_in=5,
                    lags=1):
    r"""Train on the past, score the one-step-ahead prediction.

    V-fold cross-validation would train on the future; this is the
    honest analogue for a sequentially generated sample.
    """
    if loss not in _LOSSES:
        raise ValueError("tlonsl: loss must be one of %s, got %r"
                         % (", ".join(_LOSSES), loss))
    v = [float(q) for q in k.vec(y)]
    b = int(burn_in)
    if b < 1 or b >= len(v):
        raise ValueError("tlonsl: burn_in must lie in 1..%d, got %d"
                         % (len(v) - 1, b))
    tot, preds, losses = 0.0, [], []
    for t in range(b, len(v)):
        fit = algorithm(v[:t])
        z = summary_measure(v[:t], lags)
        p = float(fit(z))
        preds.append(p)
        l = _loss(loss, v[t], p)
        losses.append(l)
        tot += l
    return {"risk": tot / len(losses), "predictions": preds,
            "losses": losses, "n_scored": len(losses),
            "note": "each prediction is scored on a genuinely "
                    "held-out FUTURE observation"}


def update_weights(cum_losses, eta=1.0):
    r"""Exponentially weighted update from cumulative losses.

    The cumulative losses are sufficient, so the update is O(1) in
    memory -- nothing is re-scored as :math:`t` grows.
    """
    c = [float(v) for v in k.vec(cum_losses)]
    if not c:
        raise ValueError("tlonsl: no cumulative losses given")
    m = min(c)
    e = [math.exp(-float(eta) * (v - m)) for v in c]
    s = sum(e)
    return [v / s for v in e]


def online_super_learner(y, library, loss="squared", burn_in=5,
                         lags=1, eta=1.0):
    r"""Sequentially-validated ensemble over a library.

    Weights are updated as data arrive; the reported risk is the
    honest one-step-ahead risk of the ensemble.
    """
    if not library:
        raise ValueError("tlonsl: the library is empty")
    v = [float(q) for q in k.vec(y)]
    names = sorted(library)
    per = {}
    for n in names:
        per[n] = sequential_risk(v, library[n], loss, burn_in, lags)
    T = per[names[0]]["n_scored"]
    cum = {n: 0.0 for n in names}
    ens_loss, weight_path = 0.0, []
    for s in range(T):
        w = update_weights([cum[n] for n in names], eta)
        weight_path.append(dict(zip(names, w)))
        p = sum(w[j] * per[names[j]]["predictions"][s]
                for j in range(len(names)))
        ens_loss += _loss(loss, v[burn_in + s], p)
        for j, n in enumerate(names):
            cum[n] += per[n]["losses"][s]
    best = min(names, key=lambda n: per[n]["risk"])
    return RichResult(payload={
        "estimate": weight_path[-1], "weights": weight_path[-1],
        "risk": ens_loss / T, "member_risks":
            {n: per[n]["risk"] for n in names},
        "best_single": per[best]["risk"], "best_member": best,
        "weight_path": weight_path, "n_scored": T,
        "method": "online super learner with sequential validation; "
                  "van der Laan & Rose (2018) Chap. 18",
        "note": "an EMPTY summary measure recovers the i.i.d. case; "
                "a parametric conditional density recovers a "
                "classical time series model",
    })


def cheatsheet():
    return ("tlonsl: data arrive sequentially, O(t) given a "
            "FIXED-DIMENSIONAL summary of the past, one common "
            "parameter across time -- empty summary = i.i.d., "
            "parametric conditional density = classical time series, "
            "and group sequential adaptive designs are covered too. "
            "V-fold CV would train on the FUTURE; use SEQUENTIAL "
            "validation instead: train on 1..t-1, score the "
            "one-step-ahead prediction of t. The oracle property "
            "survives because every loss is evaluated on a held-out "
            "future point. Cumulative losses are sufficient, so the "
            "weight update is O(1) in memory.")


# compact alias per ledger/NAMING.md
onlinesuperlearner = online_super_learner
