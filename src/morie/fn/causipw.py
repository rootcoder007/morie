# SPDX-License-Identifier: AGPL-3.0-or-later
"""IPW average treatment effect on the Crump-trimmed overlap sample."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causipw", "causal_ipw_truncated"]


def _crump_alpha(ps):
    # Empirical version of Crump et al. Corollary 5.1: alpha = 0 if
    # sup 1/(e(1-e)) <= 2 E[1/(e(1-e))]; otherwise 1/(alpha(1-alpha))
    # = gamma solves gamma = 2 E[1/(e(1-e)) | 1/(e(1-e)) <= gamma].
    k = sorted(float(1.0 / (e * (1.0 - e))) for e in ps)
    n = len(k)
    if k[-1] <= 2.0 * sum(k) / n:
        return 0.0
    best = None
    csum = 0.0
    for j in range(n):
        csum += k[j]
        gamma = 2.0 * csum / (j + 1)
        if k[j] <= gamma:
            best = gamma
    if best is None or best < 4.0:
        # gamma < 4 has no real alpha; keep everything.
        return 0.0
    disc = 0.25 - 1.0 / best
    return 0.5 - float(np.sqrt(disc))


def causipw(treat, y, ps, alpha=0.1):
    """
    Inverse-probability-weighted ATE after discarding units with
    extreme propensity scores (Crump, Hotz, Imbens and Mitnik).

    Crump et al. propose changing the estimand to the average effect on
    the overlap subpopulation A = set of x with alpha <= e(x) <=
    1 - alpha: units with estimated propensity score outside
    [alpha, 1 - alpha] are discarded, with 0.1 as the recommended
    rule-of-thumb cutoff (their Section 5 shows the fixed 0.1 rule is
    within 4 percent of the optimal-variance cutoff across their Beta
    designs). With ``alpha=None`` the optimal cutoff of Corollary 5.1
    (homoskedastic case) is computed from the scores: alpha = 0 when
    sup 1/(e(1-e)) <= 2 E[1/(e(1-e))], otherwise
    1/(alpha(1-alpha)) = 2 E[1/(e(1-e)) | 1/(e(1-e)) <= 1/(alpha(1-alpha))].

    On the retained sample the ATE is estimated by the ratio (Hajek)
    IPW form with weights normalised to sum to one within each arm,

        tau = sum(T Y / e) / sum(T / e)
              - sum((1-T) Y / (1-e)) / sum((1-T) / (1-e)).

    No analytic standard error is reported: Crump et al. treat
    inference for the trimmed estimand with standard methods on the
    selected subsample, so bootstrap or sandwich estimation on the
    retained units is left to the caller.

    Parameters
    ----------
    treat : array-like
        Binary treatment indicator, 0/1.
    y : array-like
        Outcome.
    ps : array-like
        Estimated propensity scores in (0, 1).
    alpha : float or None
        Trimming cutoff in [0, 0.5); default 0.1 (rule of thumb).
        ``None`` uses the Corollary 5.1 optimal cutoff.

    Returns
    -------
    result : RichResult
        Keys: estimate (Hajek IPW ATE on the retained sample), alpha,
        n, n_kept, n_treat_kept, n_control_kept, mean_treated,
        mean_control.

    References
    ----------
    Crump, R. K., Hotz, V. J., Imbens, G. W. and Mitnik, O. A. (2009),
    "Dealing with limited overlap in estimation of average treatment
    effects", Biometrika 96(1), 187-199, doi:10.1093/biomet/asn055.
    Working-paper version: NBER Technical Working Paper 330 (2006),
    "Moving the goalposts: Addressing limited overlap in the
    estimation of average treatment effects by changing the estimand";
    Corollary 5.1 (optimal cutoff), Section 5 (0.1 rule of thumb).
    Local source: /run/media/rootcoder/WD_BLACK/library/pdf/
    fetched-wave3/crump-hotz-imbens-mitnik-2009-limited-overlap-biometrika.pdf
    """
    t = np.asarray(treat, dtype=float)
    yv = np.asarray(y, dtype=float)
    e = np.asarray(ps, dtype=float)
    n = len(t)
    if len(yv) != n or len(e) != n:
        raise ValueError("treat, y, ps must have equal length")
    for v in t:
        if float(v) not in (0.0, 1.0):
            raise ValueError("treat must be binary 0/1")
    for v in e:
        if not 0.0 < float(v) < 1.0:
            raise ValueError("propensity scores must lie strictly in (0, 1)")
    if alpha is None:
        alpha = _crump_alpha(e)
    alpha = float(alpha)
    if not 0.0 <= alpha < 0.5:
        raise ValueError("alpha must lie in [0, 0.5)")
    keep = [i for i in range(n)
            if alpha <= float(e[i]) <= 1.0 - alpha]
    k1 = [i for i in keep if t[i] == 1.0]
    k0 = [i for i in keep if t[i] == 0.0]
    if not k1 or not k0:
        raise ValueError("trimming removed an entire treatment arm")
    w1 = [1.0 / float(e[i]) for i in k1]
    w0 = [1.0 / (1.0 - float(e[i])) for i in k0]
    mu1 = sum(w * float(yv[i]) for w, i in zip(w1, k1)) / sum(w1)
    mu0 = sum(w * float(yv[i]) for w, i in zip(w0, k0)) / sum(w0)
    return RichResult(payload={
        "estimate": mu1 - mu0,
        "alpha": alpha,
        "n": n,
        "n_kept": len(keep),
        "n_treat_kept": len(k1),
        "n_control_kept": len(k0),
        "mean_treated": mu1,
        "mean_control": mu0,
        "method": "Crump et al. (2009) overlap trimming + Hajek IPW",
    })


causal_ipw_truncated = causipw


def cheatsheet():
    return "causipw(treat, y, ps, alpha) -> Hajek IPW ATE on the Crump-trimmed overlap sample."
