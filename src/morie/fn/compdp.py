# morie.fn -- function file (rootcoder007/morie)
"""Basic and advanced composition of privacy budgets."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["basic_composition"]


def basic_composition(epsilons, delta_prime=None, deltas=None):
    r"""Total privacy cost of running several mechanisms on the same data.

    **Basic composition** is additive: :math:`k` mechanisms with budgets
    :math:`\varepsilon_i` are jointly
    :math:`\left(\sum_i \varepsilon_i\right)`-DP. Simple, always valid, and
    pessimistic.

    **Advanced composition** exploits that the losses partly cancel. For
    :math:`k` mechanisms each :math:`\varepsilon`-DP and any
    :math:`\delta' > 0`, the composition is
    :math:`(\tilde\varepsilon, k\delta + \delta')`-DP with

    .. math::
        \tilde\varepsilon = \sqrt{2k\ln(1/\delta')}\,\varepsilon
                            + k\varepsilon(e^{\varepsilon}-1),

    which grows like :math:`\sqrt k` rather than :math:`k` -- the reason a
    thousand-query workload is feasible at all. It is only a win once
    :math:`k` is reasonably large and :math:`\varepsilon` small; for a handful
    of queries basic composition is tighter, and this reports both so the
    smaller can be used honestly.

    Composition is the part practitioners most often skip. Every query against
    the same data spends budget, including the ones that were exploratory and
    the ones whose output was discarded.

    Parameters
    ----------
    epsilons : array-like
        Per-mechanism budgets, all positive.
    delta_prime : float, optional
        Slack for advanced composition, in (0, 1). Required for it.
    deltas : array-like, optional
        Per-mechanism deltas; default zeros.

    Returns
    -------
    RichResult
        ``basic_epsilon``, ``advanced_epsilon``, ``total_delta``,
        ``recommended``, ``k``.

    References
    ----------
    Dwork, C., Rothblum, G. N., & Vadhan, S. (2010). Boosting and differential
        privacy. *FOCS 2010*, 51-60.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    Basic composition simply adds.

    >>> r = basic_composition([0.1, 0.2, 0.3])
    >>> float(round(r["basic_epsilon"], 10))
    0.6

    Advanced composition beats it once k is large.

    >>> big = basic_composition([0.01] * 1000, delta_prime=1e-6)
    >>> bool(big["advanced_epsilon"] < big["basic_epsilon"])
    True
    >>> str(big["recommended"])
    'advanced'

    For few queries the basic bound is tighter, and that is what is
    recommended -- advanced composition is not free.

    >>> few = basic_composition([0.5] * 3, delta_prime=1e-6)
    >>> str(few["recommended"])
    'basic'

    >>> basic_composition([0.1, -0.2])
    Traceback (most recent call last):
        ...
    ValueError: every epsilon must be positive
    """
    eps = np.atleast_1d(np.asarray(epsilons, dtype=float)).ravel()
    if eps.size == 0:
        raise ValueError("epsilons must be non-empty")
    if np.any(eps <= 0) or not np.all(np.isfinite(eps)):
        raise ValueError("every epsilon must be positive")
    k = int(eps.size)
    dl = (np.zeros(k) if deltas is None
          else np.atleast_1d(np.asarray(deltas, dtype=float)).ravel())
    if dl.size != k:
        raise ValueError(f"deltas has {dl.size} entries but epsilons has {k}")
    basic = float(eps.sum())
    adv = np.nan
    if delta_prime is not None:
        if not 0.0 < delta_prime < 1.0:
            raise ValueError("delta_prime must be in (0, 1)")
        e = float(eps.max())
        adv = float(np.sqrt(2 * k * np.log(1.0 / delta_prime)) * e
                    + k * e * (np.exp(e) - 1.0))
    total_delta = float(dl.sum() + (delta_prime or 0.0))
    rec = "basic" if (np.isnan(adv) or basic <= adv) else "advanced"
    return RichResult(
        title="Privacy composition",
        summary_lines=[("k", k), ("basic epsilon", basic),
                       ("advanced epsilon", adv), ("recommended", rec)],
        payload={
            "basic_epsilon": basic, "advanced_epsilon": adv,
            "total_delta": total_delta, "recommended": rec,
            "epsilon": basic if rec == "basic" else adv,
            "k": k, "delta_prime": delta_prime,
            "method": "basic_composition",
        },
    )


def cheatsheet():
    return "compdp: basic adds (k), advanced grows as sqrt(k) -- advanced only wins for large k, small eps"
