# morie.fn -- function file (rootcoder007/morie)
"""The instrumental-variable estimand under a causal DAG."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_iv_instrumental_dag"]


def causal_iv_instrumental_dag(y, D, Z, homogeneous=False):
    r"""The Wald estimator, presented as an identification result
    about a graph rather than as a formula.

    .. math:: \hat\beta_{IV} = \frac{E[Y|Z=1]-E[Y|Z=0]}
                                    {E[D|Z=1]-E[D|Z=0]} .

    Arithmetically this is identical to
    :func:`morie.fn.causivla.causal_iv_late`, and the two agree to
    machine precision on the same data -- a fact the tests assert,
    because a difference between them would mean one is wrong. What
    differs is the ASSUMPTION SET under which the number means
    something, and therefore what it is a number *about*:

    * ``Z -> D -> Y`` with no arrow ``Z -> Y`` (EXCLUSION) and no
      common cause of ``Z`` and ``Y`` (EXCHANGEABILITY / instrument
      independence), plus ``Z`` genuinely moving ``D``
      (RELEVANCE). These three are what make ``Z`` an instrument in
      the graph.
    * With a CONSTANT treatment effect, that graph identifies the
      average treatment effect itself.
    * Without constant effects, it identifies the compliers' effect
      instead, and the LATE framing with its monotonicity condition
      is the honest description.

    ``homogeneous`` records which claim is being made, because the
    estimate is the same either way and only the interpretation
    changes. Defaulting it to ``False`` means the weaker, safer
    reading is the one that comes out unless a constant effect is
    asserted deliberately.

    None of exclusion, exchangeability or homogeneity is testable
    here. Relevance is, and is reported.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    d : array-like of 0/1, shape (n,)
        Treatment.
    z : array-like of 0/1, shape (n,)
        Instrument.
    homogeneous : bool, default False
        Assert a constant treatment effect, under which the estimand
        is the ATE rather than the compliers' effect.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``estimand``, ``relevance``,
        ``relevance_p``, ``assumptions``, ``testable``,
        ``untestable``, ``n``, ``method``.

    References
    ----------
    Imbens, G. W. and Angrist, J. D. (1994), *Econometrica*
    62:467-475. Angrist, Imbens and Rubin (1996), *JASA*
    91:444-455, for the graph-and-assumptions presentation.
    """
    from scipy import stats

    from .causivla import causal_iv_late

    o = causal_iv_late(y, D, Z)
    Dv = np.asarray(D, dtype=float).ravel()
    Zv = np.asarray(Z, dtype=float).ravel()
    z1, z0 = Zv == 1, Zv == 0
    n1, n0 = int(z1.sum()), int(z0.sum())
    sd = np.sqrt(Dv[z1].var(ddof=1) / n1 + Dv[z0].var(ddof=1) / n0)
    tstat = o["first_stage"] / sd if sd > 0 else np.inf
    return RichResult(payload={
        "beta": o["late"], "se": o["se"],
        "estimand": ("the average treatment effect, under the asserted "
                     "constant effect" if homogeneous else
                     "the compliers' average effect; NOT the population ATE "
                     "unless effects are constant"),
        "homogeneous_asserted": bool(homogeneous),
        "relevance": o["first_stage"],
        "relevance_t": float(tstat),
        "relevance_p": float(2 * stats.norm.sf(abs(tstat))),
        "assumptions": {
            "relevance": "Z moves D",
            "exclusion": "no arrow Z -> Y except through D",
            "exchangeability": "no common cause of Z and Y",
            "homogeneity_or_monotonicity":
                "constant effects gives the ATE; otherwise monotonicity "
                "gives the compliers' effect"},
        "testable": ["relevance"],
        "untestable": ["exclusion", "exchangeability",
                       "homogeneity", "monotonicity"],
        "same_number_as_late": "identical arithmetic to causal_iv_late; only "
                               "the assumption set and hence the estimand "
                               "differ",
        "n": o["n"],
        "method": "Wald / IV estimator under a Z -> D -> Y graph"})


def cheatsheet():
    return "causinst: same arithmetic as LATE -- the assumptions decide what the number means"
