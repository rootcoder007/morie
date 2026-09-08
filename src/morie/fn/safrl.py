r"""Constrained Policy Optimization for CMDPs.

Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017) "Constrained
Policy Optimization", *ICML*, arXiv:1705.10528. The CMDP framework
itself is Altman, E. (1999) *Constrained Markov Decision Processes*.

A **CMDP** is an MDP augmented with auxiliary cost functions
:math:`C_1, \dots, C_m` and limits :math:`d_1, \dots, d_m`. Writing
:math:`J_{C_i}(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_t \gamma^t
C_i(s_t, a_t, s_{t+1})]` for the :math:`C_i`-return, the feasible set
is

.. math:: \Pi_C = \{\pi \in \Pi : \forall i,\ J_{C_i}(\pi) \le d_i\},
          \qquad \pi^* = \arg\max_{\pi \in \Pi_C} J(\pi).

The point of CPO is that the constraint holds *throughout training*,
not just at the optimum -- it is the first policy-search algorithm for
CMDPs to guarantee that for arbitrary policy classes.

The theoretically justified update (eq. 10) maximises the surrogate
advantage inside a KL trust region while constraining an *upper bound*
on each cost return. For small steps that is well approximated by
linearising objective and constraints and taking a second-order
expansion of the KL, giving the convex program (eq. 11):

.. math:: \theta_{k+1} = \arg\max_\theta\ g^T(\theta - \theta_k)
          \quad\text{s.t.}\quad
          c_i + b_i^T(\theta - \theta_k) \le 0,\quad
          \tfrac12 (\theta - \theta_k)^T H (\theta - \theta_k) \le \delta,

with :math:`g` the objective gradient, :math:`b_i` the gradient of
constraint :math:`i`, :math:`H` the Fisher information matrix, and
:math:`c_i = J_{C_i}(\pi_k) - d_i` -- the current *violation*, positive
when the constraint is already breached. Its dual (eq. 12) is a convex
program in :math:`m+1` variables,

.. math:: \max_{\lambda \ge 0,\ \nu \succeq 0}\
          \frac{-1}{2\lambda}\big(g^T H^{-1} g - 2 r^T \nu
          + \nu^T S \nu\big) + \nu^T c - \frac{\lambda \delta}{2},
          \qquad r = g^T H^{-1} B,\quad S = B^T H^{-1} B,

and the primal solution follows from eq. 13:

.. math:: \theta^* = \theta_k + \frac{1}{\lambda^*}
          H^{-1}\big(g - B \nu^*\big).

With no constraints this collapses to the natural-gradient / TRPO step
:math:`\theta_k + \sqrt{2\delta / (g^T H^{-1} g)}\, H^{-1} g`, which is
the cleanest available check on the algebra and is anchored as such.

Infeasibility is a real case, not an edge case: approximation error can
leave :math:`\pi_k` outside :math:`\Pi_C` with no feasible step inside
the trust region. Section 6.2 then replaces the update with a
**recovery** step that purely decreases the constraint value,

.. math:: \theta^* = \theta_k - \sqrt{\frac{2\delta}{b^T H^{-1} b}}\,
          H^{-1} b,

which is what ``recovery`` in the returned result flags.

Proposition 2 bounds the damage when the linearisation is imperfect:

.. math:: J_{C_i}(\pi_{k+1}) \le d_i +
          \frac{\sqrt{2\delta}\,\gamma\, \epsilon^{\pi_{k+1}}_{C_i}}
               {(1-\gamma)^2},

so the constraint can be exceeded, but only by a bounded amount that
shrinks with the trust region. :func:`worst_case_violation` computes
it.

This module provides the CMDP machinery and the CPO step itself, given
:math:`g`, :math:`B`, :math:`c` and :math:`H` -- the quantities an
outer loop estimates from rollouts. :func:`cmdp_returns` computes
:math:`J` and :math:`J_{C_i}` for a tabular CMDP so the whole thing can
be exercised, and :func:`lagrangian_cmdp` solves a small tabular CMDP
exactly by the linear program the paper cites as the known-model
solution -- the reference the approximate step should agree with.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["safrl", "safe_rl", "cpo_step", "cmdp_returns",
           "worst_case_violation"]


def _mat(M, name):
    rows = [[float(v) for v in r]
            for r in np.atleast_2d(np.asarray(M, dtype=float))]
    if not rows or not rows[0]:
        raise ValueError("safrl: %s must be non-empty" % name)
    w = len(rows[0])
    for r in rows:
        if len(r) != w:
            raise ValueError("safrl: %s must be rectangular" % name)
    return rows


def _vec(v, name):
    out = [float(x) for x in np.atleast_1d(np.asarray(v, dtype=float))]
    if not out:
        raise ValueError("safrl: %s must be non-empty" % name)
    return out


def _solve(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError("safrl: H is singular; CPO assumes the Fisher "
                             "information matrix is positive definite")
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f == 0.0:
                continue
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def safrl(g, H, B=None, c=None, delta=0.01, tol=1e-12, max_iter=5000):
    r"""One CPO step: solve eq. 11 through its dual and return eq. 13.

    Parameters
    ----------
    g : array-like
        Objective gradient :math:`g` (of the surrogate advantage).
    H : array-like
        Fisher information matrix, assumed positive definite as the
        paper does.
    B : array-like, optional
        Constraint gradients as columns, :math:`B = [b_1, \dots, b_m]`,
        shape ``(n, m)``. Omit for the unconstrained case, which
        reduces to the natural-gradient step.
    c : array-like, optional
        :math:`c_i = J_{C_i}(\pi_k) - d_i`, the current violations.
        Positive means the constraint is already breached.
    delta : float
        Trust-region radius :math:`\delta`.
    tol, max_iter : float, int
        Controls for the dual solve.

    Returns
    -------
    RichResult
        ``estimate`` / ``step`` is :math:`\theta^* - \theta_k`;
        ``lambda_`` and ``nu`` the dual solution; ``feasible`` whether
        a feasible step existed inside the trust region;
        ``recovery`` whether the section 6.2 recovery step was taken
        instead; ``predicted_gain`` :math:`g^T \Delta\theta`;
        ``predicted_violation`` :math:`c_i + b_i^T \Delta\theta`; and
        ``kl`` the second-order KL of the step, which must not exceed
        :math:`\delta`.

    References
    ----------
    Achiam, Held, Tamar & Abbeel (2017) arXiv:1705.10528, eqs. 10-14,
    Proposition 2, Algorithm 1.
    """
    gv = _vec(g, "g")
    Hm = _mat(H, "H")
    n = len(gv)
    if len(Hm) != n or len(Hm[0]) != n:
        raise ValueError("safrl: H must be (n, n) matching g")
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("safrl: delta must be > 0")

    Hinv_g = _solve(Hm, gv)
    q = sum(gv[i] * Hinv_g[i] for i in range(n))     # g^T H^-1 g
    if q < 0.0:
        raise ValueError("safrl: g^T H^-1 g < 0; H is not positive definite")

    if B is None or c is None:
        # Unconstrained: the natural gradient step of TRPO.
        scale = math.sqrt(2.0 * delta / q) if q > 0 else 0.0
        step = [scale * v for v in Hinv_g]
        return _finish(step, gv, [], [], Hm, delta, None, [], True, False)

    Bm = _mat(B, "B")
    if len(Bm) != n:
        raise ValueError("safrl: B must have one row per parameter")
    m = len(Bm[0])
    cv = _vec(c, "c")
    if len(cv) != m:
        raise ValueError("safrl: c must have one entry per constraint")

    cols = [[Bm[i][j] for i in range(n)] for j in range(m)]
    Hinv_b = [_solve(Hm, col) for col in cols]
    r = [sum(gv[i] * Hinv_b[j][i] for i in range(n)) for j in range(m)]
    S = [[sum(cols[a][i] * Hinv_b[b][i] for i in range(n))
          for b in range(m)] for a in range(m)]

    # Feasibility of eq. 11: for a single constraint the trust region
    # can satisfy c + b^T dtheta <= 0 iff c <= sqrt(2 delta b^T H^-1 b),
    # since the most negative attainable b^T dtheta is exactly that.
    feasible = True
    for j in range(m):
        reach = math.sqrt(max(0.0, 2.0 * delta * S[j][j]))
        if cv[j] > reach + 1e-12:
            feasible = False
            break

    if not feasible:
        # Section 6.2 recovery: move purely to reduce the violated
        # constraint, as far as the trust region allows.
        j = max(range(m), key=lambda k: cv[k])
        denom = S[j][j]
        if denom <= 0.0:
            raise ValueError("safrl: constraint %d has zero curvature; "
                             "cannot recover" % j)
        scale = math.sqrt(2.0 * delta / denom)
        step = [-scale * v for v in Hinv_b[j]]
        return _finish(step, gv, cols, cv, Hm, delta, None, [], False, True)

    lam, nu = _dual(q, r, S, cv, delta, m, tol, max_iter)
    # eq. 13. Written as sqrt(2 delta / A) H^-1 (g - B nu) rather than
    # dividing by lambda, which is the same number but stays finite
    # when the constraints pin the step to zero and A -> 0 with it.
    Bnu = [sum(cols[j][i] * nu[j] for j in range(m)) for i in range(n)]
    rhs = [gv[i] - Bnu[i] for i in range(n)]
    Hinv_rhs = _solve(Hm, rhs)
    A = sum(rhs[i] * Hinv_rhs[i] for i in range(n))
    if A <= 1e-12 * max(1.0, q):
        # lambda* = 0: the objective gradient is entirely absorbed by
        # the constraint multipliers (g = B nu), so the KL constraint is
        # INACTIVE and eq. 13's division by lambda does not apply. The
        # primal optimum is then the zero step -- every direction the
        # objective wants is blocked by an active constraint. Scaling
        # the (numerically arbitrary) residual up to the full trust
        # region here would spend the whole KL budget going nowhere
        # useful, and would violate the constraints it just satisfied.
        step = [0.0] * n
        lam = 0.0
    else:
        scale = math.sqrt(2.0 * delta / A)
        step = [scale * v for v in Hinv_rhs]
    return _finish(step, gv, cols, cv, Hm, delta, lam, nu, True, False)


def _dual(q, r, S, c, delta, m, tol, max_iter):
    r"""Solve eq. 12 for :math:`\lambda^*, \nu^*`.

    Eliminating :math:`\lambda` analytically: with
    :math:`A(\nu) = (g - B\nu)^T H^{-1} (g - B\nu)
    = q - 2r^T\nu + \nu^T S \nu`, the dual function is
    :math:`A/(2\lambda) - \nu^T c + \lambda\delta`, minimised over
    :math:`\lambda > 0` at

    .. math:: \lambda^*(\nu) = \sqrt{A(\nu) / (2\delta)},

    leaving :math:`\max_{\nu \succeq 0}\ \nu^T c
    - \sqrt{2\delta A(\nu)}`.

    The factor of two is load-bearing: eq. 11's trust region is
    :math:`\tfrac12 \Delta\theta^T H \Delta\theta \le \delta`, so
    with no constraints this must reproduce the natural-gradient step
    :math:`\sqrt{2\delta / q}\,H^{-1}g` and spend exactly
    :math:`\delta` of KL -- which ``sqrt(A/delta)`` would not.

    Solved by projected gradient ascent with a backtracking line
    search on the objective itself, because the gradient blows up as
    :math:`A \to 0` (which is exactly what happens when the
    constraints pin the step to zero).
    """
    nu = [0.0] * m

    def A_of(v):
        a = q
        for j in range(m):
            a -= 2.0 * r[j] * v[j]
        for j in range(m):
            if v[j] == 0.0:
                continue
            for k in range(m):
                a += v[j] * S[j][k] * v[k]
        return max(a, 0.0)

    def obj(v):
        return sum(v[j] * c[j] for j in range(m)) - math.sqrt(
            2.0 * delta * A_of(v))

    cur = obj(nu)
    lr = 1.0
    for _ in range(int(max_iter)):
        A = A_of(nu)
        lam = math.sqrt(A / (2.0 * delta)) if A > 0 else 0.0
        if lam <= 1e-14:
            break
        grad = []
        for j in range(m):
            dA = -2.0 * r[j] + 2.0 * sum(S[j][k] * nu[k]
                                         for k in range(m))
            grad.append(c[j] - dA / (2.0 * lam))
        if max(abs(v) for v in grad) < tol:
            break
        step = lr
        improved = False
        for _bt in range(60):
            cand = [max(0.0, nu[j] + step * grad[j]) for j in range(m)]
            val = obj(cand)
            if val > cur + 1e-18:
                nu = cand
                cur = val
                improved = True
                lr = min(lr * 1.5, 1e6)
                break
            step *= 0.5
        if not improved:
            break
    A = A_of(nu)
    lam = math.sqrt(A / (2.0 * delta)) if A > 0 else 0.0
    return lam, nu


def _finish(step, g, cols, c, H, delta, lam, nu, feasible, recovery):
    n = len(g)
    kl = 0.0
    for i in range(n):
        for j in range(n):
            kl += 0.5 * step[i] * H[i][j] * step[j]
    viol = [c[j] + sum(cols[j][i] * step[i] for i in range(n))
            for j in range(len(cols))]
    return RichResult(payload={
        "estimate": step,
        "step": step,
        "lambda_": lam,
        "nu": nu,
        "feasible": bool(feasible),
        "recovery": bool(recovery),
        "predicted_gain": float(sum(g[i] * step[i] for i in range(n))),
        "predicted_violation": viol,
        "kl": float(kl),
        "delta": float(delta),
        "method": "CPO step (Achiam et al. 2017, eqs. 11-14)",
    })


def cmdp_returns(policy, states, actions, step, reward, costs, gamma=0.9,
                 start=None, iters=5000, tol=1e-14):
    r"""Exact :math:`J(\pi)` and :math:`J_{C_i}(\pi)` for a tabular CMDP.

    ``costs`` is a sequence of callables ``C_i(s, a, s_next)``. Returns
    the reward return and one constraint return per cost, evaluated by
    policy evaluation on the known model -- the quantity the feasible
    set :math:`\Pi_C` is defined by.
    """
    S = list(states)
    A = list(actions)
    dists = [reward] + list(costs)
    out = []
    for fn in dists:
        V = dict((s, 0.0) for s in S)
        for _ in range(int(iters)):
            new = {}
            for s in S:
                tot = 0.0
                for a in A:
                    p = policy(s, a)
                    if p == 0.0:
                        continue
                    s1 = step(s, a)
                    tot += p * (fn(s, a, s1) + gamma * V[s1])
                new[s] = tot
            if max(abs(new[s] - V[s]) for s in S) < tol:
                V = new
                break
            V = new
        if start is None:
            out.append(sum(V[s] for s in S) / len(S))
        elif callable(start):
            out.append(sum(start(s) * V[s] for s in S))
        else:
            out.append(V[start])
    return RichResult(payload={
        "estimate": out[0],
        "J": out[0],
        "J_C": out[1:],
        "gamma": float(gamma),
        "method": "CMDP returns (Altman 1999; Achiam et al. 2017 sec. 4)",
    })


def worst_case_violation(delta, gamma, epsilon):
    r"""Proposition 2's bound,
    :math:`\sqrt{2\delta}\,\gamma\,\epsilon / (1-\gamma)^2`.

    The amount by which :math:`J_{C_i}(\pi_{k+1})` may exceed
    :math:`d_i` after a CPO update, with
    :math:`\epsilon = \max_s |\mathbb{E}_{a \sim \pi_{k+1}}
    [A^{\pi_k}_{C_i}(s,a)]|`.
    """
    delta = float(delta)
    gamma = float(gamma)
    if delta < 0.0:
        raise ValueError("worst_case_violation: delta must be >= 0")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("worst_case_violation: gamma must lie in [0, 1)")
    return (math.sqrt(2.0 * delta) * gamma * float(epsilon)
            / (1.0 - gamma) ** 2)


def cheatsheet():
    return ("safrl: CPO (Achiam 2017). CMDP = MDP + cost returns "
            "J_Ci <= d_i (Altman 1999); Pi_C is the feasible set. "
            "Step solves max g'dtheta s.t. c_i + b_i'dtheta <= 0 and "
            "0.5 dtheta'H dtheta <= delta (eq. 11) through its dual "
            "(eq. 12), giving dtheta = H^-1(g - B nu)/lambda (eq. 13). "
            "Unconstrained it IS the natural-gradient/TRPO step. "
            "Infeasible -> sec 6.2 recovery step. Prop 2 bounds the "
            "overshoot by sqrt(2 delta) gamma eps/(1-gamma)^2.")


# compact aliases per ledger/NAMING.md
safe_rl = safrl
saferl = safrl
cpo_step = safrl
