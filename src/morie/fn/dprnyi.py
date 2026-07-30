# morie.fn -- function file (rootcoder007/morie)
"""Renyi differential privacy and its composition."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["renyi_dp_composition"]


def renyi_dp_composition(epsilons, alpha=2.0, delta=1e-5):
    r"""Compose Renyi-DP budgets and convert the total to :math:`(\varepsilon, \delta)`.

    Renyi DP measures privacy loss by the Renyi divergence of order
    :math:`\alpha` between the output distributions on neighbouring inputs.
    Its appeal is that composition is *exactly* additive in
    :math:`\varepsilon_\alpha` -- no union bound, no slack term -- so a long
    training run can be accounted tightly:

    .. math::
        \varepsilon_\alpha^{\text{total}} = \sum_i \varepsilon_\alpha^{(i)} .

    Conversion to the usual pair uses

    .. math::
        \varepsilon = \varepsilon_\alpha
            + \frac{\ln(1/\delta)}{\alpha - 1},

    which is why :math:`\alpha` is a knob: it is chosen *after* accounting, by
    sweeping and keeping whichever gives the smallest :math:`\varepsilon`.
    Fixing :math:`\alpha` up front leaves accuracy on the table.

    Parameters
    ----------
    epsilons : array-like
        Per-step RDP budgets at order ``alpha``.
    alpha : float
        Renyi order, must exceed 1.
    delta : float
        Target delta for the conversion, in (0, 1).

    Returns
    -------
    RichResult
        ``rdp_total``, ``epsilon``, ``delta``, ``alpha``, ``k``.

    References
    ----------
    Mironov, I. (2017). Renyi differential privacy. *CSF 2017*, 263-275.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    RDP composes by plain addition -- no slack.

    >>> r = renyi_dp_composition([0.01] * 100, alpha=10.0, delta=1e-5)
    >>> float(round(r["rdp_total"], 10))
    1.0

    The conversion penalty shrinks as alpha grows, so alpha is worth sweeping.

    >>> best = min(renyi_dp_composition([0.01] * 100, alpha=a, delta=1e-5)["epsilon"]
    ...            for a in (2.0, 4.0, 8.0, 16.0, 32.0))
    >>> bool(best < renyi_dp_composition([0.01] * 100, alpha=2.0,
    ...                                  delta=1e-5)["epsilon"])
    True

    >>> renyi_dp_composition([0.1], alpha=1.0)
    Traceback (most recent call last):
        ...
    ValueError: alpha must be greater than 1
    """
    alpha = float(alpha)
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    eps = np.atleast_1d(np.asarray(epsilons, dtype=float)).ravel()
    if eps.size == 0:
        raise ValueError("epsilons must be non-empty")
    if np.any(eps < 0):
        raise ValueError("RDP epsilons must be non-negative")
    total = float(eps.sum())
    conv = float(total + np.log(1.0 / delta) / (alpha - 1.0))
    return RichResult(
        title="Renyi DP composition",
        summary_lines=[("alpha", alpha), ("k", int(eps.size)),
                       ("RDP total", total), ("epsilon", conv)],
        payload={
            "rdp_total": total, "epsilon": conv, "delta": float(delta),
            "alpha": alpha, "k": int(eps.size),
            "conversion_penalty": float(np.log(1.0 / delta) / (alpha - 1.0)),
            "method": "renyi_dp_composition",
        },
    )


def cheatsheet():
    return "dprnyi: RDP composes EXACTLY additively; sweep alpha after accounting to minimise final epsilon"
