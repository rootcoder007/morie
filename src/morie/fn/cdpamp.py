# morie.fn -- function file (rootcoder007/morie)
"""Concentrated DP and its subgaussian composition."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["cdp_subgaussian_amplification"]


def cdp_subgaussian_amplification(rho, k_compositions=1, delta=1e-5):
    r"""Compose zero-concentrated DP budgets and convert to :math:`(\varepsilon, \delta)`.

    zCDP measures privacy loss by a subgaussian condition on the Renyi
    divergence: a mechanism is :math:`\rho`-zCDP if
    :math:`D_\alpha(M(D) \Vert M(D')) \le \rho\alpha` for all
    :math:`\alpha > 1`. Composition is then trivially additive,

    .. math::
        \rho_{\text{total}} = \sum_i \rho_i,

    and converts by
    :math:`\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)}`.

    The :math:`\sqrt{k}` growth of advanced composition falls out of that
    formula for free rather than being bolted on with a union bound, and the
    result is tighter -- which is why zCDP is the accounting of choice for
    long training runs.

    The Gaussian mechanism is naturally :math:`\rho`-zCDP with
    :math:`\rho = \Delta_2^2/(2\sigma^2)`, so a noise multiplier converts
    directly into a budget without going through :math:`\varepsilon` at all.

    zCDP cannot express pure :math:`\varepsilon`-DP: the Laplace mechanism has
    unbounded Renyi divergence at large :math:`\alpha` and so no finite
    :math:`\rho`. Any conversion into zCDP is therefore lossy in that
    direction.

    Parameters
    ----------
    rho : float or array-like
        Per-mechanism zCDP budgets, non-negative.
    k_compositions : int
        Repeat count when ``rho`` is scalar.
    delta : float
        Target delta for the conversion, in (0, 1).

    Returns
    -------
    RichResult
        ``rho_total``, ``epsilon``, ``delta``, ``sqrt_k_growth``,
        ``equivalent_sigma``.

    References
    ----------
    Bun, M., & Steinke, T. (2016). Concentrated differential privacy:
        Simplifications, extensions, and lower bounds. *TCC 2016-B*, 635-658.

    Examples
    --------
    zCDP composes additively, and the resulting epsilon grows like sqrt(k) --
    advanced composition without the union bound.

    >>> import numpy as np
    >>> a = cdp_subgaussian_amplification(0.01, k_compositions=100)
    >>> b = cdp_subgaussian_amplification(0.01, k_compositions=400)
    >>> float(round(a["rho_total"], 10)), float(round(b["rho_total"], 10))
    (1.0, 4.0)
    >>> bool(1.6 < b["epsilon"] / a["epsilon"] < 2.4)
    True

    A Gaussian noise multiplier maps straight to a budget.

    >>> r = cdp_subgaussian_amplification(0.5)
    >>> bool(abs(r["equivalent_sigma"] - 1.0) < 1e-9)
    True

    >>> cdp_subgaussian_amplification(-1.0)
    Traceback (most recent call last):
        ...
    ValueError: rho must be non-negative
    """
    r = np.atleast_1d(np.asarray(rho, dtype=float)).ravel()
    if np.any(r < 0):
        raise ValueError("rho must be non-negative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    k = int(k_compositions)
    if k < 1:
        raise ValueError("k_compositions must be at least 1")
    total = float(r.sum() * k) if r.size == 1 else float(r.sum())
    eps = float(total + 2.0 * np.sqrt(total * np.log(1.0 / delta)))
    return RichResult(
        title="zCDP composition",
        summary_lines=[("rho total", total), ("epsilon", eps),
                       ("delta", float(delta))],
        warnings=["zCDP cannot represent pure epsilon-DP: the Laplace "
                  "mechanism has unbounded Renyi divergence and no finite rho"],
        payload={
            "rho_total": total, "epsilon": eps, "delta": float(delta),
            "sqrt_k_growth": True,
            # Gaussian mechanism with unit L2 sensitivity: rho = 1/(2 sigma^2).
            "equivalent_sigma": float(np.sqrt(1.0 / (2.0 * total))) if total > 0 else float("inf"),
            "k": k, "method": "cdp_subgaussian_amplification",
        },
    )


def cheatsheet():
    return "cdpamp: rho adds, eps = rho + 2*sqrt(rho*log(1/delta)); sqrt(k) growth comes free, no union bound"
