# morie.fn -- function file (rootcoder007/morie)
"""Sampford's unequal-probability sampling without replacement."""

from __future__ import annotations

from ._containers import DescriptiveResult
from ._qpcore import ssum
from ._rng import random_uniform


def _draw(cum, u):
    for k, c in enumerate(cum):
        if u <= c:
            return k
    return len(cum) - 1


def sampford_design(pik, *, seed: int = 0, max_iter: int = 500) -> DescriptiveResult:
    """Sampford (1967) pi-ps sample and its exact joint inclusion probabilities.

    With target inclusion probabilities ``pi_i`` (``0 < pi_i < 1``,
    ``sum pi_i = n`` an integer) the rejective procedure draws one unit
    with probability ``pi_i / n`` and ``n - 1`` further units with
    replacement with probabilities proportional to ``pi_i / (1 - pi_i)``,
    accepting the first attempt whose ``n`` units are distinct; the
    inclusion probabilities are then exactly ``pi_i``. The joint
    inclusion probabilities follow Sampford's formula with
    ``lambda_i = p_i / (1 - n p_i)``, ``p_i = pi_i / n``:
    ``pi_ij = K_n lambda_i lambda_j sum_{t=1}^{n-1} (t + 1 - n (p_i + p_j)) L_{n-t}(ij) / n^(t-1)``,
    where ``L_m`` are the elementary symmetric functions of the
    ``lambda`` (``L_m(ij)`` without units ``i`` and ``j``) and
    ``K_n = 1 / sum_{t=1}^{n} t L_{n-t} / n^t``. Units with ``pi_i``
    within ``1e-6`` of 0 or 1 are fixed out or in. Every attempt uses
    ``n`` Philox uniforms from stream ``attempt``.

    :param pik: Inclusion probabilities summing to an integer ``n >= 2``.
    :param seed: Philox key.
    :param max_iter: Maximum rejective attempts.
    :return: DescriptiveResult; ``value`` is the 0/1 sample indicator;
        ``extra`` has ``joint`` (the ``N x N`` matrix of ``pi_ij`` with
        ``pi_i`` on the diagonal), ``attempts`` and ``n``.

    References
    ----------
    Sampford, M. R. (1967). On sampling without replacement with unequal
    probabilities of selection. Biometrika 54, 499-513.

    Tille, Y. (2006). Sampling Algorithms. Springer, Sec. 7.6.

    Examples
    --------
    >>> r = sampford_design([0.2, 0.4, 0.6, 0.8])
    >>> sum(r.value), [round(t, 12) for t in r.extra["joint"][0]]
    (2, [0.2, 0.027722772277, 0.053465346535, 0.118811881188])
    """
    pik = [float(t) for t in pik]
    n = round(ssum(pik))
    if abs(ssum(pik) - n) > 1e-8 or n < 2:
        raise ValueError("pik must sum to an integer n >= 2")
    N = len(pik)
    eps = 1e-6
    free = [i for i in range(N) if eps < pik[i] < 1 - eps]
    s = [1 if pik[i] >= 1 - eps else 0 for i in range(N)]
    pb = [pik[i] for i in free]
    nb = round(ssum(pb))
    tot = ssum(pb)
    c1, acc = [], 0.0
    for t in pb:
        acc += t / tot
        c1.append(acc)
    odds = [t / (1 - t) for t in pb]
    so = ssum(odds)
    c2, acc = [], 0.0
    for t in odds:
        acc += t / so
        c2.append(acc)
    attempts, ok = 0, False
    while attempts < max_iter and not ok:
        u = [float(v) for v in random_uniform(nb, seed=seed, stream=attempts)]
        attempts += 1
        picks = [_draw(c1, u[0])] + [_draw(c2, v) for v in u[1:]]
        ok = len(set(picks)) == nb
    if not ok:
        raise RuntimeError("too many rejective attempts")
    for k in picks:
        s[free[k]] = 1
    joint = [[0.0] * N for _ in range(N)]
    Jb = (
        _sampford_joint(pb, nb)
        if nb >= 2
        else [[pb[a] if a == b else 0.0 for b in range(len(pb))] for a in range(len(pb))]
    )
    for a, i in enumerate(free):
        for b, j in enumerate(free):
            joint[i][j] = Jb[a][b]
    for i in range(N):
        if s[i] and i not in free:
            for j in range(N):
                joint[i][j] = joint[j][i] = pik[j] if j in free else float(s[j])
    return DescriptiveResult(name="sampford_design", value=s, extra={"joint": joint, "attempts": attempts, "n": n})


def _sampford_joint(pik, n):
    N = len(pik)
    p = [t / n for t in pik]
    lam = [t / (1 - n * t) for t in p]
    L = [0.0] * (n + 1)
    L[1] = 1.0
    for i in range(2, n + 1):
        acc = 0.0
        for r in range(1, i):
            acc += (-1) ** (r - 1) * ssum(t**r for t in lam) * L[i - r]
        L[i] = acc / (i - 1)
    if any(t < 0 for t in L[1:]):
        raise ValueError("joint inclusion probabilities cannot be computed for these pik")
    Kn = 1.0 / ssum((n + 1 - m) * L[m] / n ** (n + 1 - m) for m in range(1, n + 1))
    P = [[0.0] * N for _ in range(N)]
    for i in range(N):
        P[i][i] = pik[i]
        for j in range(i):
            L2 = [0.0] * n
            L2[1] = 1.0
            if n > 2:
                L2[2] = L[2] - (lam[i] + lam[j])
            for m in range(3, n):
                L2[m] = L[m] - (lam[i] + lam[j]) * L2[m - 1] - lam[i] * lam[j] * L2[m - 2]
            v = Kn * lam[i] * lam[j] * ssum((t + 1 - n * (p[i] + p[j])) * L2[n - t] / n ** (t - 1) for t in range(1, n))
            P[i][j] = P[j][i] = v
    return P


sampfd = sampford_design


def cheatsheet() -> str:
    return "sampford_design(pik) -> Sampford pi-ps sample without replacement and exact joint inclusion probabilities"
