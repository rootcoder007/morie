# morie.fn -- function file (rootcoder007/morie)
"""i.i.d. contraction theorem in Hellinger metric: main tool for density estimation rates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_iid_crt_thm"]


def ghosal_iid_crt_thm(x, eps=None, n=None, prior_mass=None, entropy=None,
                       C=1.0):
    r"""The i.i.d. posterior contraction theorem (Ghosal Sec. 8.2).

    A rate :math:`\varepsilon_n` is attained,

    .. math:: \Pi\big(p : d_H(p, p_0) > M\varepsilon_n \mid X_1..X_n\big)
              \to 0,

    when THREE conditions hold together:

    1. **entropy** -- the model has
       :math:`\log N(\varepsilon_n, \mathcal P_n, d_H)
       \lesssim n\varepsilon_n^2`;
    2. **prior mass** -- the prior charges a KL neighbourhood,
       :math:`\Pi(B_{KL}(p_0,\varepsilon_n)) \ge
       e^{-Cn\varepsilon_n^2}`;
    3. **sieve remainder** -- the prior mass outside the sieve is
       :math:`o(e^{-(C+4)n\varepsilon_n^2})`.

    All three are calibrated by the SAME quantity
    :math:`n\varepsilon_n^2`, and that is the content: the rate is
    whatever makes entropy and prior mass balance. Neither alone
    gives a rate. A prior can put ample mass near :math:`p_0` and
    still fail if the model is too rich to test in, and a small
    model buys nothing if the prior starves the truth.

    Hellinger distance is not incidental either -- it is bounded and
    tensorises, so a test at :math:`\varepsilon` for one observation
    gives an exponentially powerful test for n, which is what the
    proof runs on.

    Parameters
    ----------
    x : array-like
        Observations; used for the sample size.
    eps : float, optional
        Candidate rate to check; ``n^{-1/3}`` otherwise.
    n : int, optional
        Sample size.
    prior_mass : float, optional
        Measured :math:`\Pi(B_{KL})`, if available.
    entropy : float, optional
        Measured :math:`\log N(\varepsilon_n)`, if available.
    C : float
        The constant in the prior-mass condition.

    Returns
    -------
    RichResult
        keys: ``n``, ``eps``, ``n_eps_squared``,
        ``entropy_budget``, ``prior_mass_budget``,
        ``entropy_ok``, ``prior_mass_ok``, ``all_conditions_checked``,
        ``metric`` ("Hellinger"), ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 8.2 (and 8.2.1-8.2.2);
    Ghosal, Ghosh and van der Vaart (2000).
    """
    xv = np.asarray(x, dtype=float).ravel()
    nn = int(xv.size) if n is None else int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    e = float(nn ** (-1.0 / 3.0)) if eps is None else float(eps)
    if e <= 0:
        raise ValueError(f"eps must be positive, got {e}.")
    ne2 = nn * e * e
    ent_budget = ne2
    pm_budget = float(np.exp(-float(C) * ne2))
    ent_ok = None if entropy is None else bool(float(entropy) <= ent_budget)
    pm_ok = None if prior_mass is None else \
        bool(float(prior_mass) >= pm_budget)
    return RichResult(payload={
        "n": nn, "eps": e, "n_eps_squared": float(ne2),
        "entropy_budget": float(ent_budget),
        "prior_mass_budget": pm_budget,
        "entropy_ok": ent_ok, "prior_mass_ok": pm_ok,
        "all_conditions_checked": bool(ent_ok is not None and pm_ok is not None),
        "metric": "Hellinger",
        "conditions": ("entropy <= n eps^2; prior mass >= exp(-C n eps^2); "
                       "sieve remainder o(exp(-(C+4) n eps^2))"),
        "method": "i.i.d. contraction theorem (Sec. 8.2); all three conditions calibrated by n eps^2"})


def cheatsheet():
    return "gh_c8_6: entropy and prior mass must BALANCE at n eps^2 -- neither alone gives a rate"
