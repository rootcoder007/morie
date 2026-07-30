# morie.fn -- function file (rootcoder007/morie)
"""Rosenbaum sensitivity bounds."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_rosenbaum_bound"]


def causal_rosenbaum_bound(paired_diff, gamma_max=3.0, n_gamma=25, alpha=0.05):
    r"""How much hidden bias would overturn a matched-pair result.

    Under no hidden bias, matched units have equal treatment odds. Rosenbaum's
    :math:`\Gamma` bounds the departure:

    .. math::
        \frac{1}{\Gamma} \le
        \frac{\pi_i(1-\pi_j)}{\pi_j(1-\pi_i)} \le \Gamma

    for matched :math:`i, j`. At :math:`\Gamma = 1` the design is a
    randomised experiment; at :math:`\Gamma = 2`, one unit of a matched pair
    could be twice as likely to be treated for reasons never measured. The
    signed-rank test is evaluated at its worst case over that range, and the
    reported :math:`\Gamma^*` is where significance is lost.

    This does **not** test whether hidden bias exists -- nothing in the data
    can. It converts a result into a statement of the form "an unmeasured
    confounder would have to move treatment odds by a factor of
    :math:`\Gamma^*` to explain this away", which is a claim a reader can
    weigh against what they know about the setting.

    A small :math:`\Gamma^*` is not a refutation and a large one is not a
    proof. The number is a unit of fragility, and it is only meaningful next
    to the plausible confounders in the specific application.

    Parameters
    ----------
    paired_diff : array-like
        Within-pair outcome differences (treated minus control).
    gamma_max : float
        Largest :math:`\Gamma` to scan.
    n_gamma : int
        Grid points.
    alpha : float
        Significance level.

    Returns
    -------
    RichResult
        ``gamma_critical``, ``gamma_grid``, ``p_upper``,
        ``significant_at_gamma_1``, ``interpretation``.

    References
    ----------
    Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.

    Examples
    --------
    A strong effect survives substantial hidden bias.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> d = rng.normal(2.0, 1.0, 200)
    >>> r = causal_rosenbaum_bound(d)
    >>> bool(r["gamma_critical"] > 2.0)
    True

    A marginal one does not: it breaks almost immediately.

    >>> weak = rng.normal(0.15, 1.0, 200)
    >>> bool(causal_rosenbaum_bound(weak)["gamma_critical"] < 1.5)
    True

    Gamma = 1 is the randomised-experiment case, where the bound is the
    ordinary signed-rank p-value.

    >>> bool(r["p_upper"][0] < 0.001)
    True

    >>> causal_rosenbaum_bound(np.arange(1.0, 11.0), gamma_max=0.5)
    Traceback (most recent call last):
        ...
    ValueError: gamma_max must be at least 1
    """
    from scipy.stats import norm

    d = np.atleast_1d(np.asarray(paired_diff, dtype=float)).ravel()
    d = d[d != 0]
    n = d.size
    if n < 5:
        raise ValueError("need at least 5 non-zero pair differences")
    if gamma_max < 1:
        raise ValueError("gamma_max must be at least 1")

    ranks = np.argsort(np.argsort(np.abs(d), kind="stable"), kind="stable") + 1.0
    w_plus = float(np.sum(ranks[d > 0]))
    total = n * (n + 1) / 2.0
    grid = np.linspace(1.0, float(gamma_max), int(n_gamma))
    p_up = np.empty(grid.size)
    for i, g in enumerate(grid):
        # Worst case: each pair's positive probability pushed to g/(1+g).
        p = g / (1.0 + g)
        mu = p * total
        var = p * (1 - p) * float(np.sum(ranks**2))
        z = (w_plus - mu) / np.sqrt(max(var, 1e-300))
        p_up[i] = float(norm.sf(z))
    sig = p_up < alpha
    gcrit = float(grid[sig][-1]) if sig.any() else 1.0
    return RichResult(
        title="Rosenbaum sensitivity bound",
        summary_lines=[("pairs", int(n)), ("Gamma*", gcrit),
                       ("p at Gamma=1", float(p_up[0]))],
        warnings=["this does not test whether hidden bias exists; it states "
                  "how large it would have to be, and is only meaningful "
                  "against the plausible confounders in the application"],
        payload={
            "gamma_critical": gcrit, "gamma_grid": grid, "p_upper": p_up,
            "significant_at_gamma_1": bool(p_up[0] < alpha),
            "interpretation": (
                f"an unmeasured confounder would need to change treatment odds "
                f"by a factor of {gcrit:.2f} within matched pairs to overturn "
                "this result"),
            "n_pairs": int(n), "alpha": float(alpha),
            "method": "causal_rosenbaum_bound",
        },
    )


def cheatsheet():
    return "causrosen: converts a result into 'how much hidden bias would kill it'; not a test that bias exists"
