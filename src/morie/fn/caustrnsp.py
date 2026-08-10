# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transport/generalize a trial effect to a target population by weighting."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["caustrnsp", "causal_transport_weights"]


def caustrnsp(y, z, s, mode="transport", pr_w0=None):
    """
    Weighting estimator of the population average treatment effect
    that generalizes or transports a randomized-trial effect to a
    target population, using estimated sampling (participation)
    scores.

    Identification of transported effects is due to Pearl and
    Bareinboim (2014); the estimator implemented here is the weighting
    form given by Tipton and Hartman (2023, Eq. 3.10): with W = 1 for
    trial units, Z the randomized treatment and w_i the weights,

        tau_w = sum(w T Y) / sum(w T) - sum(w (1-T) Y) / sum(w (1-T)),

    the weights normalised to sum to one within each arm. For
    generalizability (sample a subset of the target) the weights are
    inverse sampling probabilities w_i = 1 / s_i (Eq. 3.11); for
    transport to a disjoint target population they are the odds
    w_i = (1 - s_i) / s_i divided by Pr(W = 0) (Eq. 3.12; the
    normalisation by Pr(W = 0) cancels in the ratio form and is
    accepted for completeness).

    Parameters
    ----------
    y : array-like
        Outcomes of the trial units.
    z : array-like
        Randomized binary treatment of the trial units, 0/1.
    s : array-like
        Estimated sampling scores s(x) of the trial units, strictly in
        (0, 1).
    mode : str
        "transport" (odds weights, Eq. 3.12) or "generalize" (inverse
        probability weights, Eq. 3.11).
    pr_w0 : float, optional
        Pr(W = 0) for the unnormalised Eq. 3.12 weights; only affects
        the reported weights, not the estimate (it cancels).

    Returns
    -------
    result : RichResult
        Keys: estimate (weighted PATE), mean_treated, mean_control,
        weights, n, n_treat, n_control, mode.

    References
    ----------
    Tipton, E. and Hartman, E. (2023), "Generalizability and
    transportability", Ch. 3 in Zubizarreta, Stuart, Small, Rosenbaum
    (eds), Handbook of Matching and Weighting Adjustments for Causal
    Inference, Chapman and Hall/CRC, doi:10.1201/9781003102670,
    Eqs. 3.10-3.12. Local source: /run/media/rootcoder/WD_BLACK/
    library/pdf/ (Zubizarreta et al. Handbook PDF), Ch. 3.
    Pearl, J. and Bareinboim, E. (2014), "External validity: From
    do-calculus to transportability across populations", Statistical
    Science 29(4), 579-595, doi:10.1214/14-STS486. Local copy:
    fetched-wave3/pearl-bareinboim-2014-external-validity-transportability-StatSci29.pdf
    """
    yv = np.asarray(y, dtype=float)
    zv = np.asarray(z, dtype=float)
    sv = np.asarray(s, dtype=float)
    n = len(yv)
    if len(zv) != n or len(sv) != n:
        raise ValueError("y, z, s must have equal length")
    for v in zv:
        if float(v) not in (0.0, 1.0):
            raise ValueError("z must be binary 0/1")
    for v in sv:
        if not 0.0 < float(v) < 1.0:
            raise ValueError("sampling scores must lie strictly in (0, 1)")
    if mode not in ("transport", "generalize"):
        raise ValueError("mode must be 'transport' or 'generalize'")
    if mode == "generalize":
        w = [1.0 / float(v) for v in sv]
    else:
        c = float(pr_w0) if pr_w0 is not None else 1.0
        if not c > 0.0:
            raise ValueError("pr_w0 must be positive")
        w = [(1.0 - float(v)) / float(v) / c for v in sv]
    i1 = [i for i in range(n) if zv[i] == 1.0]
    i0 = [i for i in range(n) if zv[i] == 0.0]
    if not i1 or not i0:
        raise ValueError("need both treatment arms in the trial sample")
    mu1 = sum(w[i] * float(yv[i]) for i in i1) / sum(w[i] for i in i1)
    mu0 = sum(w[i] * float(yv[i]) for i in i0) / sum(w[i] for i in i0)
    return RichResult(payload={
        "estimate": mu1 - mu0,
        "mean_treated": mu1,
        "mean_control": mu0,
        "weights": np.asarray(w),
        "n": n, "n_treat": len(i1), "n_control": len(i0),
        "mode": mode,
        "method": "Tipton-Hartman Eq. 3.10 weighted PATE, %s weights" % mode,
    })


causal_transport_weights = caustrnsp


def cheatsheet():
    return "caustrnsp(y, z, s, mode) -> trial effect generalized/transported to a target population."
