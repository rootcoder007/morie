# morie.fn -- function file (rootcoder007/morie)
"""Privacy amplification by subsampling."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["privacy_amplification"]


def privacy_amplification(epsilon, q, delta=0.0):
    r"""Effective budget when a mechanism sees only a random ``q``-fraction.

    Running an :math:`\varepsilon`-DP mechanism on a Poisson subsample of rate
    :math:`q` gives

    .. math::
        \varepsilon' = \ln\!\left(1 + q\left(e^{\varepsilon} - 1\right)\right),

    and :math:`\delta' = q\delta`. For small :math:`\varepsilon` this is
    approximately :math:`q\varepsilon` -- sampling 1% of records buys roughly a
    hundred-fold reduction in privacy cost, which is the whole reason DP-SGD
    on minibatches is affordable.

    The guarantee depends on the sampling being **secret and fresh**. Reusing
    a fixed subsample, or letting an adversary learn who was sampled, destroys
    the amplification entirely -- the record is either in or out, and there is
    no uncertainty left to hide behind. Shuffling or a fixed held-out split
    does not qualify.

    Parameters
    ----------
    epsilon : float
        Budget of the base mechanism, positive.
    q : float
        Sampling rate in (0, 1].
    delta : float
        Base delta; scales by ``q``.

    Returns
    -------
    RichResult
        ``epsilon_amplified``, ``delta_amplified``, ``ratio``,
        ``linear_approx``.

    References
    ----------
    Balle, B., Barthe, G., & Gaboardi, M. (2018). Privacy amplification by
        subsampling. *NeurIPS 2018*.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-407.

    Examples
    --------
    Sampling reduces the effective budget, close to linearly for small
    epsilon.

    >>> r = privacy_amplification(epsilon=0.1, q=0.01)
    >>> bool(r["epsilon_amplified"] < 0.1)
    True
    >>> bool(abs(r["epsilon_amplified"] - r["linear_approx"]) < 1e-4)
    True

    Sampling everything is a no-op, as it must be.

    >>> float(round(privacy_amplification(0.5, q=1.0)["epsilon_amplified"], 10))
    0.5

    >>> privacy_amplification(0.1, q=0.0)
    Traceback (most recent call last):
        ...
    ValueError: q must be in (0, 1]
    """
    epsilon, _ = check_budget(epsilon)
    q = float(q)
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    eps_a = float(np.log1p(q * np.expm1(epsilon)))
    return RichResult(
        title="Privacy amplification by subsampling",
        summary_lines=[("epsilon", epsilon), ("q", q),
                       ("amplified epsilon", eps_a)],
        warnings=["amplification holds only for secret, freshly drawn "
                  "subsamples; a fixed or observable subsample gives none"],
        payload={
            "epsilon_amplified": eps_a, "delta_amplified": float(q * delta),
            "ratio": float(eps_a / epsilon), "linear_approx": float(q * epsilon),
            "epsilon": epsilon, "q": q, "method": "privacy_amplification",
        },
    )


def cheatsheet():
    return "dpamp: eps' = log(1 + q(e^eps - 1)) ~ q*eps; requires SECRET fresh sampling or it is void"
