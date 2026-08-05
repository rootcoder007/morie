# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a multi-arm treatment: arm means and pairwise contrasts."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_multiple_treatments"]


def tmle_multiple_treatments(y, D, X, arm_set):
    """Targeted mean outcome under each arm, plus contrasts to the first.

    With more than two arms the propensity is a full multinomial, and a
    single clever covariate can no longer solve every arm's score.  The
    fluctuation is therefore saturated in the arm: one ``eps_a`` per arm
    with clever covariate ``H_a = I(A = a) / g_a(X)``, which is the
    same thing as fluctuating along the vector-valued clever covariate
    and is what makes every arm mean, not just one contrast, a valid
    plug-in.

    The multinomial is built from one-versus-rest logistics normalised to
    sum to one across the arm set.  That is a working model, not the
    multinomial MLE; it is used because it is deterministic, mirrors
    exactly across the two language arms, and the targeting step repairs
    the bias it introduces as long as either it or the outcome model is
    right.

        ``psi_a = mean_i [Q(a, X_i) + eps_a / g_a(X_i)]``,
        ``eps_a = sum_i H_a (y_i - Q(A_i, X_i)) / sum_i H_a^2``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Arm label of each subject; every value must appear in ``arm_set``.
    X : array-like, shape (n, p)
        Covariates.
    arm_set : array-like, shape (k,)
        The distinct arm labels, in the reporting order.  The first entry
        is the reference for the contrasts.

    Returns
    -------
    RichResult
        ``estimate`` (last arm minus first arm), ``se``, ``psi`` (list of
        arm means), ``contrasts`` (list, each arm minus the first),
        ``n_arms``, ``n``.

    References
    ----------
    Lendle, S. D., Schwab, J., Petersen, M. L. & van der Laan, M. J.
    (2017).  ltmle: an R package implementing targeted minimum
    loss-based estimation for longitudinal data.  Journal of Statistical
    Software 81(1).  doi:10.18637/jss.v081.i01.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    arms = C.vec(arm_set)
    n = len(yv)
    k = len(arms)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_multiple_treatments: y and D must share one length")
    if k < 2:
        raise ValueError("tmle_multiple_treatments: need at least two arms")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_multiple_treatments: X must have one row per subject")
    for i in range(n):
        if not any(abs(Dv[i] - a) < 1e-9 for a in arms):
            raise ValueError("tmle_multiple_treatments: an arm label is not in arm_set")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    ind = [[1.0 if abs(Dv[i] - arms[j]) < 1e-9 else 0.0 for j in range(k)] for i in range(n)]

    raw = []
    for j in range(k):
        b = S.glmbin(W, [ind[i][j] for i in range(n)])
        raw.append([S.clip(S.expit(C.dot(W[i], b)), 0.01, 0.99) for i in range(n)])
    g = [[0.0] * k for _ in range(n)]
    for i in range(n):
        tot = sum(raw[j][i] for j in range(k))
        for j in range(k):
            g[i][j] = S.clip(raw[j][i] / tot, 0.01, 0.99)

    des = [list(W[i]) + [ind[i][j] for j in range(1, k)] for i in range(n)]
    qb, _, _, _ = S.ols(des, yv)

    def qhat(i, j):
        row = list(W[i]) + [1.0 if j == m else 0.0 for m in range(1, k)]
        return C.dot(row, qb)

    Qobs = [C.dot(des[i], qb) for i in range(n)]
    psi = []
    ics = []
    for j in range(k):
        H = [ind[i][j] / g[i][j] for i in range(n)]
        den = sum(h * h for h in H)
        e = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
        Qs = [qhat(i, j) + e / g[i][j] for i in range(n)]
        p = sum(Qs) / n
        psi.append(p)
        ics.append([H[i] * (yv[i] - Qobs[i] - e * H[i]) + Qs[i] - p for i in range(n)])
    contrasts = [psi[j] - psi[0] for j in range(k)]
    ic = [ics[k - 1][i] - ics[0][i] for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": contrasts[k - 1], "se": se, "psi": psi,
        "contrasts": contrasts, "n_arms": float(k), "n": n,
        "method": "TMLE for a multi-arm treatment with pairwise contrasts"})


def cheatsheet():
    return "tmlmlt: TMLE for multi-arm treatments and their contrasts."
