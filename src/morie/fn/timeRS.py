# morie.fn -- function file (rootcoder007/morie)
r"""timeSVD++: preferences drift, and drift is not noise.

Two temporal effects in the Netflix data motivate the whole model, and
both are measured rather than posited: the mean rating **jumped** from
about 3.4 to above 3.6 stars in early 2004, and ratings **rise with a
movie's age** -- older films score higher than newer ones.

**Why the usual remedies fail here.** Concept-drift work tracks a
single concept; here many characteristics of many users and items
shift at once. A time window or instance decay would discard the older
data, and the paper is explicit that this loses too many signals. The
alternative is to *model* the drift and keep every instance.

**Different effects drift at different rates, so they get different
treatments.**

* **Item bias** moves slowly -- a film's perception changes over
  months -- so it is captured with **time bins**:
  :math:`b_i(t) = b_i + b_{i,\mathrm{Bin}(t)}`.
* **User bias** can shift gradually and also spike on a single day, so
  it gets a smooth term plus a per-day term:
  :math:`b_u(t) = b_u + \alpha_u\,\mathrm{dev}_u(t) + b_{u,t}`.

**The deviation function is the mechanism.** With :math:`t_u` the
user's mean rating date,

.. math:: \mathrm{dev}_u(t) = \mathrm{sign}(t - t_u)\,
          |t - t_u|^{\beta},\qquad \beta = 0.4,

so the effect is **signed** -- before and after the user's centre pull
in opposite directions -- and **concave**, so a rating 400 days out is
not four times the deviation of one 100 days out. :math:`\beta` was
set by cross-validation, not derived, and ``deviation`` exposes it so
the concavity can be checked instead of assumed.

**A single-day term is not overfitting, it is the point.** Ratings
made in one session share a mood; :math:`b_{u,t}` absorbs that so it
does not contaminate the long-run parameters.

References
----------
Koren, Y. (2010) "Collaborative filtering with temporal dynamics",
*Communications of the ACM* 53(4), 89-97,
doi:10.1145/1721654.1721677. [PDF supplied by Vee.] The two measured
effects in the Netflix data -- the abrupt shift of rating scale in
early 2004 from around 3.4 to above 3.6 stars, and ratings increasing
with movie age; the argument that this differs from concept drift
because many characteristics shift simultaneously and that classical
time-window or instance-decay approaches cannot work as they lose too
many signals when discarding data instances; the time deviation
dev_u(t) = sign(t - t_u) |t - t_u|^beta with t_u the user's mean
rating date and beta = 0.4 set by cross-validation; the resulting
time-dependent user bias; item bias in time bins; and the per-day user
term.

Koren, Y. (2009) "Collaborative Filtering with Temporal Dynamics",
*KDD '09*, 447-456, doi:10.1145/1557019.1557072. The full conference
treatment, including the spline alternative to the linear drift.

Koren, Y. (2008) "Factorization Meets the Neighborhood", *KDD '08*,
426-434, doi:10.1145/1401890.1401944. The SVD++ base this extends;
implemented in :mod:`svdpp`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["deviation", "time_bin", "user_bias", "item_bias",
           "predict_time", "fit_time_bias"]

_EPS = 1e-12
BETA = 0.4


def deviation(t, t_user, beta=BETA):
    r""":math:`\mathrm{sign}(t-t_u)|t-t_u|^{\beta}`, :math:`\beta=0.4`.

    Signed, so the two sides of the user's centre pull oppositely, and
    concave, so distant ratings are not weighted linearly.
    """
    d = float(t) - float(t_user)
    b = float(beta)
    if b <= 0.0:
        raise ValueError("timeRS: beta must be positive")
    return (1.0 if d > 0 else (-1.0 if d < 0 else 0.0)) \
        * abs(d) ** b


def time_bin(t, bin_days=70, n_bins=30):
    r"""Which slow-moving bin a date falls in.

    Item perception changes over months, so a bin is the right
    granularity -- a per-day item term would have almost no data in
    it.
    """
    w = int(bin_days)
    if w < 1:
        raise ValueError("timeRS: the bin width must be positive")
    return min(int(max(float(t), 0.0) // w), int(n_bins) - 1)


def user_bias(b_u, alpha_u, t, t_user, per_day=None, beta=BETA):
    r""":math:`b_u + \alpha_u\mathrm{dev}_u(t) + b_{u,t}`."""
    dev = deviation(t, t_user, beta)
    day = 0.0 if per_day is None else float(per_day.get(int(t), 0.0))
    return {"bias": float(b_u) + float(alpha_u) * dev + day,
            "deviation": dev, "per_day": day,
            "note": "a single-day term absorbs session mood, so it "
                    "does not contaminate the long-run parameters"}


def item_bias(b_i, bins, t, bin_days=70, n_bins=30):
    r""":math:`b_i + b_{i,\mathrm{Bin}(t)}`."""
    idx = time_bin(t, bin_days, n_bins)
    return {"bias": float(b_i) + float(bins[idx])
            if idx < len(bins) else float(b_i), "bin": idx}


def predict_time(mu, b_u, alpha_u, t_user, b_i, item_bins, t,
                 p_u=None, q_i=None, per_day=None, bin_days=70,
                 beta=BETA):
    r"""The time-aware prediction."""
    ub = user_bias(b_u, alpha_u, t, t_user, per_day, beta)
    ib = item_bias(b_i, item_bins, t, bin_days, len(item_bins))
    inner = 0.0
    if p_u is not None and q_i is not None:
        p = [float(v) for v in k.vec(p_u)]
        q = [float(v) for v in k.vec(q_i)]
        if len(p) != len(q):
            raise ValueError("timeRS: the factors differ in width")
        inner = sum(p[a] * q[a] for a in range(len(p)))
    return {"prediction": float(mu) + ub["bias"] + ib["bias"] + inner,
            "user_bias": ub["bias"], "item_bias": ib["bias"],
            "deviation": ub["deviation"], "bin": ib["bin"]}


def fit_time_bias(ratings, n_users, n_items, bin_days=70,
                  n_bins=30, epochs=40, lr=0.005, reg=0.02,
                  beta=BETA):
    r"""Fit the time-dependent biases by SGD.

    ``ratings`` are ``(user, item, day, value)``. Every instance is
    kept -- the alternative the paper rejects is discarding old ones.
    """
    R = [(int(u), int(i), float(t), float(r))
         for u, i, t, r in ratings]
    if not R:
        raise ValueError("timeRS: no ratings given")
    nu, ni = int(n_users), int(n_items)
    mu = sum(r for _, _, _, r in R) / len(R)
    days = {}
    for u, _, t, _ in R:
        days.setdefault(u, []).append(t)
    t_user = {u: sum(v) / len(v) for u, v in days.items()}
    bu = [0.0] * nu
    al = [0.0] * nu
    bi = [0.0] * ni
    bins = [[0.0] * int(n_bins) for _ in range(ni)]
    hist = []
    for _ in range(int(epochs)):
        se = 0.0
        for (u, i, t, r) in R:
            dev = deviation(t, t_user[u], beta)
            idx = time_bin(t, bin_days, n_bins)
            pred = mu + bu[u] + al[u] * dev + bi[i] + bins[i][idx]
            e = r - pred
            se += e * e
            bu[u] += lr * (e - reg * bu[u])
            al[u] += lr * (e * dev - reg * al[u])
            bi[i] += lr * (e - reg * bi[i])
            bins[i][idx] += lr * (e - reg * bins[i][idx])
        hist.append(math.sqrt(se / len(R)))
    return RichResult(payload={
        "estimate": hist[-1], "rmse": hist[-1], "rmse_history": hist,
        "mu": mu, "b_user": bu, "alpha_user": al, "b_item": bi,
        "item_bins": bins, "t_user": t_user, "beta": float(beta),
        "n_instances": len(R),
        "method": "time-dependent biases; Koren (2010) eq. (8)",
        "note": "every instance is kept; windows and decay would "
                "discard the signals this models",
    })


def cheatsheet():
    return ("timeRS: preferences DRIFT -- the Netflix mean rating "
            "jumped 3.4 to 3.6 in early 2004 and ratings rise with "
            "movie age. This is not ordinary concept drift (many "
            "things shift at once), and windows or decay would discard "
            "too much, so MODEL the drift and keep every instance. "
            "Different effects, different rates: item bias in slow "
            "TIME BINS (months), user bias smooth PLUS a per-day term "
            "for session mood. The mechanism is dev_u(t) = "
            "sign(t - t_u)|t - t_u|^0.4 -- SIGNED, so the two sides of "
            "the user's centre pull oppositely, and CONCAVE, so 400 "
            "days out is not 4x 100 days out. beta was cross-"
            "validated, not derived.")


# compact alias per ledger/NAMING.md
timesvdpp = fit_time_bias
