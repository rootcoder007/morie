"""Hierarchical mixture of experts, two levels, fitted by EM (ESL sec 9.5)."""

import math

from ._richresult import RichResult
from .linsys import _householder_ls
from .nlsgn import _inverse

__all__ = ["esl_hme"]


def _log_gate(coef, z, K):
    """Log-softmax gate from esl_multinomial_logit coefficients (last class baseline)."""
    eta = [sum(c * v for c, v in zip(row, z)) for row in coef] + [0.0]
    m = max(eta)
    lse = m + math.log(sum(math.exp(v - m) for v in eta))
    return [v - lse for v in eta]


def _fit_gate(Z, W, coef, iters=25):
    """Multinomial-logit gate on soft targets: rows Z, target weights W (N x K), last class baseline.

    Newton steps from the warm start ``coef`` with step halving; a 1e-8 ridge on
    the information matrix keeps saturated gates invertible.
    """
    N, q, K = len(Z), len(Z[0]), len(W[0])
    npar = (K - 1) * q
    b = [v for row in coef for v in row]

    def probs(bb):
        out = []
        for z in Z:
            eta = [sum(bb[k * q + j] * z[j] for j in range(q)) for k in range(K - 1)] + [0.0]
            m = max(eta)
            e = [math.exp(v - m) for v in eta]
            s = sum(e)
            out.append([v / s for v in e])
        return out

    def obj(P):
        return sum(W[i][k] * math.log(max(P[i][k], 1e-300)) for i in range(N) for k in range(K) if W[i][k] > 0)

    P = probs(b)
    cur = obj(P)
    wt = [sum(r) for r in W]
    for _ in range(iters):
        grad = [sum(Z[i][a % q] * (W[i][a // q] - wt[i] * P[i][a // q]) for i in range(N)) for a in range(npar)]
        info = [
            [
                sum(
                    wt[i] * Z[i][a % q] * Z[i][c % q] * P[i][a // q] * ((a // q == c // q) - P[i][c // q])
                    for i in range(N)
                )
                + (1e-8 if a == c else 0.0)
                for c in range(npar)
            ]
            for a in range(npar)
        ]
        inv = _inverse(info)
        step = [sum(inv[a][c] * grad[c] for c in range(npar)) for a in range(npar)]
        fac = 1.0
        while fac >= 1 / 1024:
            cand = [x + fac * d for x, d in zip(b, step)]
            Pc = probs(cand)
            oc = obj(Pc)
            if oc >= cur - 1e-12:
                b, P, cur = cand, Pc, oc
                break
            fac /= 2
        if max(abs(fac * d) for d in step) < 1e-10:
            break
    return [[b[k * q + j] for j in range(q)] for k in range(K - 1)]


def _wls(Z, y, w):
    sw = [math.sqrt(max(v, 0.0)) for v in w]
    return _householder_ls([[s * v for v in r] for s, r in zip(sw, Z)], [s * v for s, v in zip(sw, y)])[0]


def _wlogit(Z, y, w, beta, iters=50):
    """Weighted logistic regression by IRLS (expert of eq 9.29)."""
    b = beta[:]
    for _ in range(iters):
        eta = [sum(c * v for c, v in zip(b, z)) for z in Z]
        p = [1 / (1 + math.exp(-min(max(e, -30), 30))) for e in eta]
        ww = [wi * max(pi * (1 - pi), 1e-10) for wi, pi in zip(w, p)]
        zz = [e + (yi - pi) / max(pi * (1 - pi), 1e-10) for e, yi, pi in zip(eta, y, p)]
        nb = _wls(Z, zz, ww)
        if max(abs(a - c) for a, c in zip(nb, b)) < 1e-10:
            return nb
        b = nb
    return b


def esl_hme(X, y, K=2, task="regression", max_iter=500, tol=1e-10):
    r"""Two-level hierarchical mixture of experts fitted by EM.

    ESL eqs 9.25-9.30: top gate :math:`g_j(x) = e^{\gamma_j^Tx}/\sum_k e^{\gamma_k^Tx}`,
    second-level gates :math:`g_{\ell|j}(x)`, and experts
    :math:`\Pr(y|x,\theta_{j\ell})`: Gaussian linear regressions
    :math:`y = \beta_{j\ell}^Tx + \epsilon`, :math:`\epsilon\sim N(0,\sigma^2_{j\ell})` (eq 9.28)
    or logistic regressions (eq 9.29). The likelihood
    :math:`\Pr(y|x) = \sum_jg_j(x)\sum_\ell g_{\ell|j}(x)\Pr(y|x,\theta_{j\ell})` (eq 9.30)
    is maximised by EM: the E-step gives the branch posteriors; the M-step
    fits each expert by weighted least squares (or weighted IRLS) and each
    gate as a multinomial logit on the posterior weights (Newton steps with a
    1e-8 ridge that keeps saturated gates invertible; Jordan & Jacobs 1994). An intercept is added to x; the start assigns the observations to
    the K^2 experts in (softened) blocks of the first input.

    Parameters
    ----------
    X : N x p nested sequence
    y : sequence of N floats (0/1 for task="classification")
    K : int
        Branching factor at each level.
    task : {"regression", "classification"}
    max_iter, tol
        EM controls (relative change in the log-likelihood).

    Returns
    -------
    RichResult
        ``loglik``, ``loglik_path``, ``experts`` (coefficients per (j, l), index
        j K + l), ``sigma2`` (regression), ``top_gate`` and ``sub_gates``
        (gate coefficients, last branch as baseline), ``fitted``,
        ``iterations``, ``converged``.

    References
    ----------
    Jordan, M. I. & Jacobs, R. A. (1994). Hierarchical mixtures of experts and
    the EM algorithm. Neural Computation 6, 181-214.
    """
    rows = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    N = len(rows)
    if task not in ("regression", "classification") or len(yy) != N or K < 2:
        raise ValueError("need matching X and y, K >= 2 and task 'regression' or 'classification'")
    Z = [[1.0] + r for r in rows]
    q = len(Z[0])
    order = sorted(range(N), key=lambda i: (rows[i][0], i))
    H = [[0.0] * (K * K) for _ in range(N)]
    for pos, i in enumerate(order):
        # softened block start: a hard 0/1 start makes the first gate fit separable
        blk = min(pos * K * K // N, K * K - 1)
        H[i] = [0.9 if e == blk else 0.1 / (K * K - 1) for e in range(K * K)]
    top = [[0.0] * q for _ in range(K - 1)]
    sub = [[[0.0] * q for _ in range(K - 1)] for _ in range(K)]
    beta = [[0.0] * q for _ in range(K * K)]
    s2 = [1.0] * (K * K)

    def mstep(H):
        nonlocal top, sub, beta, s2
        for e in range(K * K):
            w = [H[i][e] for i in range(N)]
            if task == "regression":
                beta[e] = _wls(Z, yy, w)
                r2 = sum(wi * (yi - sum(c * v for c, v in zip(beta[e], z))) ** 2 for wi, yi, z in zip(w, yy, Z))
                s2[e] = max(r2 / max(sum(w), 1e-300), 1e-12)
            else:
                beta[e] = _wlogit(Z, yy, w, beta[e])
        hj = [[sum(H[i][j * K + ell] for ell in range(K)) for j in range(K)] for i in range(N)]
        top = _fit_gate(Z, hj, top)
        for j in range(K):
            sub[j] = _fit_gate(Z, [[H[i][j * K + ell] for ell in range(K)] for i in range(N)], sub[j])

    def logdens(e, z, yi):
        m = sum(c * v for c, v in zip(beta[e], z))
        if task == "regression":
            return -0.5 * (yi - m) ** 2 / s2[e] - 0.5 * math.log(2 * math.pi * s2[e])
        x = -m if yi == 1 else m  # log sigma(+-m) = -softplus(x), computed stably
        return -(max(x, 0.0) + math.log1p(math.exp(-abs(x))))

    path, prev, conv, it = [], -math.inf, False, 0
    for _ in range(max_iter):
        it += 1
        mstep(H)
        ll = 0.0
        for i, z in enumerate(Z):
            gt = _log_gate(top, z, K)
            lj = []
            for j in range(K):
                gs = _log_gate(sub[j], z, K)
                for ell in range(K):
                    lj.append(gt[j] + gs[ell] + logdens(j * K + ell, z, yy[i]))
            m = max(lj)
            e = [math.exp(v - m) for v in lj]
            tot = sum(e)
            H[i] = [v / tot for v in e]
            ll += m + math.log(tot)
        path.append(ll)
        if abs(ll - prev) < tol * (1 + abs(ll)):
            conv = True
            break
        prev = ll
    fitted = []
    for z in Z:
        gt = _log_gate(top, z, K)
        f = 0.0
        for j in range(K):
            gs = _log_gate(sub[j], z, K)
            for ell in range(K):
                m = sum(c * v for c, v in zip(beta[j * K + ell], z))
                f += math.exp(gt[j] + gs[ell]) * (
                    m if task == "regression" else 1 / (1 + math.exp(-min(max(m, -30), 30)))
                )
        fitted.append(f)
    return RichResult(
        title="Hierarchical mixture of experts",
        summary_lines=[("loglik", ll), ("iterations", it)],
        payload={
            "loglik": ll,
            "loglik_path": path,
            "experts": beta,
            "sigma2": s2 if task == "regression" else None,
            "top_gate": top,
            "sub_gates": sub,
            "fitted": fitted,
            "iterations": it,
            "converged": conv,
        },
    )


def cheatsheet():
    return "eslhme: EM for a 2-level HME; experts by weighted LS/IRLS, gates by weighted multinomial logits (ESL 9.25-9.30)"
