# morie.fn -- function file (rootcoder007/morie)
r"""Reversible-jump MCMC: sampling across models of differing dimension.

Green, P. J. (1995) "Reversible jump Markov chain Monte Carlo computation
and Bayesian model determination", *Biometrika* 82(4), 711-732.
doi:10.1093/biomet/82.4.711

An ordinary Metropolis-Hastings sampler needs the state to keep a fixed
length. Bayesian model determination does not: the indicator :math:`k`
for the model is part of the state, and the parameter vector
:math:`\theta^{(k)}` that goes with it changes length as :math:`k`
changes. Green's construction makes that legal.

**The general case (§3-2).** The chain is a hybrid sampler: at each
transition a move type :math:`m` is chosen at random from those
available, with probability :math:`j(x)` depending on the current state.
Detailed balance is imposed within each move type. Provided
:math:`\pi(dx) q_m(x, dx')` has a density with respect to a *symmetric*
measure -- the dimension-matching Assumption -- the acceptance
probability

.. math::

   \alpha_m(x, x') = \min\left\{1,
     \frac{\pi(dx')\,q_m(x', dx)}{\pi(dx)\,q_m(x, dx')}\right\}

is the largest one satisfying balance (Peskun 1973).

**Switching between two subspaces (§3-3).** The template that makes the
Assumption concrete. To go from :math:`(1, \theta^{(1)})` to
:math:`(2, \theta^{(2)})`, draw :math:`u^{(1)}` of length :math:`m_1`
and set :math:`\theta^{(2)}` to a deterministic function of
:math:`\theta^{(1)}` and :math:`u^{(1)}`; to come back, draw
:math:`u^{(2)}` of length :math:`m_2`. **There must be a bijection
between** :math:`(\theta^{(1)}, u^{(1)})` **and**
:math:`(\theta^{(2)}, u^{(2)})`; in particular the lengths must satisfy

.. math::   n_1 + m_1 = n_2 + m_2.

That is the whole of dimension matching, and this module refuses to run
a move that violates it. Equation 7 is then

.. math::

   \min\left\{1,\;
   \frac{p(2, \theta^{(2)} \mid y)\, j(2, \theta^{(2)})\, q_2(u^{(2)})}
        {p(1, \theta^{(1)} \mid y)\, j(1, \theta^{(1)})\, q_1(u^{(1)})}
   \left| \frac{\partial(\theta^{(2)}, u^{(2)})}
               {\partial(\theta^{(1)}, u^{(1)})} \right| \right\}

-- posterior ratio, move-probability ratio, proposal-density ratio, and
the Jacobian of the bijection. When :math:`m_1` or :math:`m_2` is zero
the corresponding :math:`q` drops out and this is Equation 8, which the
paper's own applications use throughout. Note Remark 3: the priors
:math:`p(\theta^{(k)} \mid k)` may share **one** unknown normalising
constant, no more -- relative constants between subspaces are needed or
balance between subspaces cannot hold.

Two routes are provided and both are kept.

``reversible_jump_mcmc`` is the general §3 engine. The caller supplies
models (a dimension and a log posterior each) and moves (a bijection,
the proposal for :math:`u`, and its density). ``j(x)`` is computed by
the engine from the move weights available in each model, in both
directions, because Equation 7 needs it at :math:`x'` as well as at
:math:`x`. The Jacobian may be given analytically or, with
``jacobian="numeric"``, differenced from the bijection itself --
``numeric_log_jacobian`` is also public so an analytic one can be
checked against it.

``changepoint_rjmcmc`` is the paper's §4 application: a step-function
rate for a point process on :math:`[0, L]`, with the number of steps
:math:`k` free. Its four moves are Green's -- height change, position
change, birth, death -- with the acceptance ratios printed in §4-3, and
its defaults are the run reported in §4-4 (:math:`\lambda = 3`,
:math:`k_{\max} = 30`, :math:`\alpha = 1`, :math:`\beta = 200`, 40000
updates after 4000 burn-in). Passing ``use_likelihood=False`` drops the
likelihood so the chain must reproduce the prior exactly; that is the
sharpest available check that the birth/death ratio is right.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "bayrjmcmc",
    "reversible_jump_mcmc",
    "check_dimension_matching",
    "rj_log_acceptance",
    "numeric_log_jacobian",
    "changepoint_rjmcmc",
    "step_function_loglik",
    "changepoint_move_probabilities",
    "birth_split_heights",
    "birth_log_jacobian",
]

_JACOBIAN_ROUTES = ("analytic", "numeric")


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def _unif_stream(seed, block=8192):
    """Uniforms from the package Philox generator, drawn in blocks."""
    st = {"buf": [], "i": 0, "stream": 0}

    def uni():
        if st["i"] >= len(st["buf"]):
            st["buf"] = [float(v) for v in
                         random_uniform(block, seed=seed, stream=st["stream"])]
            st["stream"] += 1
            st["i"] = 0
        v = st["buf"][st["i"]]
        st["i"] += 1
        if v <= 0.0:
            v = 1e-15
        elif v >= 1.0:
            v = 1.0 - 1e-15
        return v

    return uni


def _logabsdet(a):
    """log|det A| by Gaussian elimination with partial pivoting."""
    n = len(a)
    if n == 0:
        return 0.0
    m = [list(row) for row in a]
    total = 0.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            return float("-inf")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        total += math.log(abs(m[col][col]))
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f == 0.0:
                continue
            for c in range(col, n):
                m[r][c] -= f * m[col][c]
    return total


def numeric_log_jacobian(mapfun, z, h=1e-6):
    r"""``log|det d f / d z|`` of a bijection, by central differences.

    ``mapfun`` takes a list and returns a list of the same length --
    that equality *is* the dimension-matching condition, so a mismatch
    raises rather than silently returning a rectangular derivative.
    """
    z = [float(v) for v in z]
    n = len(z)
    out0 = mapfun(z)
    if len(out0) != n:
        raise ValueError(
            "bayrjmcmc: the bijection maps %d values to %d; dimension "
            "matching requires n1 + m1 == n2 + m2" % (n, len(out0)))
    if n == 0:
        return 0.0
    jac = [[0.0] * n for _ in range(n)]
    for j in range(n):
        step = h * max(1.0, abs(z[j]))
        zp = list(z)
        zm = list(z)
        zp[j] += step
        zm[j] -= step
        fp = mapfun(zp)
        fm = mapfun(zm)
        for i in range(n):
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * step)
    return _logabsdet(jac)


# --------------------------------------------------------------------------
# the general construction, §3
# --------------------------------------------------------------------------

_MOVE_KEYS = ("frm", "to", "n_u", "n_u_rev", "propose", "transform")


def check_dimension_matching(models, moves):
    r"""Enforce the §3-3 Assumption on a set of moves.

    Every move must have a reverse; the reverse must generate exactly
    the :math:`u` the forward move consumes and vice versa; and
    :math:`n_1 + m_1 = n_2 + m_2`. Returns the ``(frm, to) -> move``
    lookup the sampler needs to evaluate :math:`j(x')`.
    """
    if not models:
        raise ValueError("bayrjmcmc: no models given")
    for name, spec in models.items():
        if "dim" not in spec or "logpost" not in spec:
            raise ValueError(
                "bayrjmcmc: model %r needs 'dim' and 'logpost'" % (name,))
        if int(spec["dim"]) < 0:
            raise ValueError("bayrjmcmc: model %r has negative dim" % (name,))
    by_pair = {}
    for mv in moves:
        for key in _MOVE_KEYS:
            if key not in mv:
                raise ValueError("bayrjmcmc: a move is missing %r" % (key,))
        if mv["frm"] not in models or mv["to"] not in models:
            raise ValueError("bayrjmcmc: move %r -> %r names an unknown model"
                             % (mv["frm"], mv["to"]))
        if mv["frm"] == mv["to"]:
            raise ValueError(
                "bayrjmcmc: %r -> %r is a within-model move; give it as "
                "'within', not as a jump" % (mv["frm"], mv["to"]))
        if (mv["frm"], mv["to"]) in by_pair:
            raise ValueError("bayrjmcmc: two moves given for %r -> %r"
                             % (mv["frm"], mv["to"]))
        by_pair[(mv["frm"], mv["to"])] = mv
    for mv in moves:
        rev = by_pair.get((mv["to"], mv["frm"]))
        if rev is None:
            raise ValueError(
                "bayrjmcmc: move %r -> %r has no reverse move; detailed "
                "balance is imposed within each move type, so the reverse "
                "must be supplied" % (mv["frm"], mv["to"]))
        n1 = int(models[mv["frm"]]["dim"])
        n2 = int(models[mv["to"]]["dim"])
        m1 = int(mv["n_u"])
        m2 = int(mv["n_u_rev"])
        if m1 < 0 or m2 < 0:
            raise ValueError("bayrjmcmc: move %r -> %r has negative n_u"
                             % (mv["frm"], mv["to"]))
        if n1 + m1 != n2 + m2:
            raise ValueError(
                "bayrjmcmc: move %r -> %r violates dimension matching: "
                "n1 + m1 = %d + %d != %d + %d = n2 + m2"
                % (mv["frm"], mv["to"], n1, m1, n2, m2))
        if int(rev["n_u"]) != m2 or int(rev["n_u_rev"]) != m1:
            raise ValueError(
                "bayrjmcmc: move %r -> %r declares u of length %d and a "
                "reverse u of length %d, but the reverse move declares "
                "%d and %d" % (mv["frm"], mv["to"], m1, m2,
                               int(rev["n_u"]), int(rev["n_u_rev"])))
    return by_pair


def rj_log_acceptance(logpost_from, logpost_to, log_j_from, log_j_to,
                      logq_u, logq_u_rev, log_jacobian):
    r"""Log of the Equation 7 ratio (before the ``min`` with 1).

    .. math::

       \log\frac{p(2,\theta^{(2)}|y)\,j(2,\theta^{(2)})\,q_2(u^{(2)})}
                {p(1,\theta^{(1)}|y)\,j(1,\theta^{(1)})\,q_1(u^{(1)})}
       + \log\left|\frac{\partial(\theta^{(2)},u^{(2)})}
                        {\partial(\theta^{(1)},u^{(1)})}\right|

    With :math:`m_1 = 0` there is no :math:`u^{(1)}` to generate, so
    ``logq_u`` is zero and this is Equation 8.
    """
    return ((logpost_to - logpost_from)
            + (log_j_to - log_j_from)
            + (logq_u_rev - logq_u)
            + log_jacobian)


def _rw_within(theta, uni, scale):
    """Symmetric Gaussian random walk; the proposal ratio is 1."""
    out = []
    for v in theta:
        u1 = uni()
        u2 = uni()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        out.append(v + scale * z)
    return out


def reversible_jump_mcmc(models, moves, init_model, init_theta=(),
                         n_iter=10000, burn_in=0, thin=1, seed=0,
                         within=None, within_scale=0.5, within_weight=1.0,
                         move_weight=1.0, jacobian="analytic",
                         keep_chain=True):
    r"""Green's §3 reversible-jump sampler over a set of models.

    Parameters
    ----------
    models : dict
        ``name -> {"dim": int, "logpost": callable}``. ``logpost(theta)``
        returns :math:`\log p(k, \theta^{(k)} \mid y)` up to **one**
        constant shared by every model (Remark 3). ``dim`` may be 0.
    moves : list of dict
        Each jump move needs ``frm``, ``to``, ``n_u``, ``n_u_rev``,
        ``propose(theta, uni) -> u``, and
        ``transform(theta, u) -> (theta2, u2)``. Optional: ``logq``
        and ``logq_rev`` (log densities of :math:`u^{(1)}` and
        :math:`u^{(2)}`, default 0 for the unit-uniform case),
        ``logjac`` (default 0 for a volume-preserving map, or
        differenced when ``jacobian="numeric"``), ``weight`` (its share
        of :math:`j(x)`, default ``move_weight``), and ``name``.
        Every move must have its reverse in the list.
    within : dict, optional
        ``name -> callable(theta, uni) -> (theta_new, log_q_ratio)`` for
        the fixed-dimension update in each model. The default is a
        symmetric Gaussian random walk of scale ``within_scale``, whose
        ratio is 1.
    jacobian : {"analytic", "numeric"}
        Where the Jacobian comes from. ``"numeric"`` differences the
        bijection, which is the honest default when the algebra has not
        been done and the right way to audit it when it has.

    Returns
    -------
    RichResult
        ``model_freq`` (posterior model probabilities from the visit
        counts), ``visits``, ``accept`` per move, ``chain`` of
        ``(model, theta)`` after burn-in and thinning.
    """
    if jacobian not in _JACOBIAN_ROUTES:
        raise ValueError("bayrjmcmc: jacobian must be one of %s"
                         % (_JACOBIAN_ROUTES,))
    by_pair = check_dimension_matching(models, moves)
    if init_model not in models:
        raise ValueError("bayrjmcmc: init_model %r is not a model"
                         % (init_model,))
    n_iter = int(n_iter)
    burn_in = int(burn_in)
    thin = int(thin)
    if n_iter < 1:
        raise ValueError("bayrjmcmc: n_iter must be at least 1")
    if burn_in < 0 or burn_in >= n_iter:
        raise ValueError("bayrjmcmc: burn_in must be in [0, n_iter)")
    if thin < 1:
        raise ValueError("bayrjmcmc: thin must be at least 1")

    theta = [float(v) for v in init_theta]
    if len(theta) != int(models[init_model]["dim"]):
        raise ValueError("bayrjmcmc: init_theta has %d values but model %r "
                         "has dim %d" % (len(theta), init_model,
                                         int(models[init_model]["dim"])))

    # j(x) depends only on the model, so tabulate it once: a within-model
    # move plus every jump leaving that model.
    avail = {}
    for name in models:
        opts = [("within", None, float(within_weight))]
        for mv in moves:
            if mv["frm"] == name:
                opts.append(("jump", mv, float(mv.get("weight", move_weight))))
        tot = sum(w for _, _, w in opts)
        if tot <= 0.0:
            raise ValueError("bayrjmcmc: model %r has no move with positive "
                             "weight" % (name,))
        avail[name] = (opts, tot)

    uni = _unif_stream(seed)
    cur = init_model
    logp = float(models[cur]["logpost"](theta))
    visits = dict((name, 0) for name in models)
    tried = {}
    accepted = {}
    chain = []

    for it in range(n_iter):
        opts, tot = avail[cur]
        pick = uni() * tot
        acc = 0.0
        kind, mv, w = opts[-1]
        for k, m, ww in opts:
            acc += ww
            if pick < acc:
                kind, mv, w = k, m, ww
                break

        if kind == "within":
            label = "within:%s" % (cur,)
            tried[label] = tried.get(label, 0) + 1
            if theta:
                if within is not None and cur in within:
                    prop, log_ratio = within[cur](theta, uni)
                    prop = [float(v) for v in prop]
                else:
                    prop = _rw_within(theta, uni, within_scale)
                    log_ratio = 0.0
                lp_new = float(models[cur]["logpost"](prop))
                if math.log(uni()) < (lp_new - logp) + log_ratio:
                    theta, logp = prop, lp_new
                    accepted[label] = accepted.get(label, 0) + 1
        else:
            label = mv.get("name") or ("%s->%s" % (mv["frm"], mv["to"]))
            tried[label] = tried.get(label, 0) + 1
            u = [float(v) for v in mv["propose"](theta, uni)]
            if len(u) != int(mv["n_u"]):
                raise ValueError(
                    "bayrjmcmc: move %s proposed %d values of u but "
                    "declares n_u = %d" % (label, len(u), int(mv["n_u"])))
            theta2, u2 = mv["transform"](theta, u)
            theta2 = [float(v) for v in theta2]
            u2 = [float(v) for v in u2]
            dim2 = int(models[mv["to"]]["dim"])
            if len(theta2) != dim2 or len(u2) != int(mv["n_u_rev"]):
                raise ValueError(
                    "bayrjmcmc: move %s produced theta of length %d and u2 "
                    "of length %d; the model has dim %d and the move "
                    "declares n_u_rev = %d" % (label, len(theta2), len(u2),
                                               dim2, int(mv["n_u_rev"])))

            if jacobian == "numeric" or "logjac" not in mv:
                if jacobian == "analytic" and "logjac" not in mv:
                    logjac = 0.0          # volume preserving by declaration
                else:
                    n_from = len(theta)
                    tf = mv["transform"]

                    def _flat(z, _tf=tf, _n=n_from):
                        a, b = _tf(z[:_n], z[_n:])
                        return [float(v) for v in a] + [float(v) for v in b]

                    logjac = numeric_log_jacobian(_flat, list(theta) + list(u))
            else:
                logjac = float(mv["logjac"](theta, u, theta2, u2))

            rev = by_pair[(mv["to"], mv["frm"])]
            opts2, tot2 = avail[mv["to"]]
            w_rev = float(rev.get("weight", move_weight))
            log_j_from = math.log(w) - math.log(tot)
            log_j_to = math.log(w_rev) - math.log(tot2)

            logq_u = (float(mv["logq"](theta, u)) if "logq" in mv else 0.0)
            logq_rev = (float(mv["logq_rev"](theta2, u2))
                        if "logq_rev" in mv else 0.0)

            lp_new = float(models[mv["to"]]["logpost"](theta2))
            log_alpha = rj_log_acceptance(logp, lp_new, log_j_from, log_j_to,
                                          logq_u, logq_rev, logjac)
            if math.log(uni()) < log_alpha:
                cur, theta, logp = mv["to"], theta2, lp_new
                accepted[label] = accepted.get(label, 0) + 1

        if it >= burn_in:
            visits[cur] += 1
            if keep_chain and ((it - burn_in) % thin == 0):
                chain.append((cur, list(theta)))

    kept = sum(visits.values())
    freq = dict((name, visits[name] / float(kept)) for name in models)
    rates = dict((k, accepted.get(k, 0) / float(v))
                 for k, v in tried.items() if v)
    return RichResult(payload={
        "model_freq": freq,
        "visits": visits,
        "n_kept": kept,
        "accept": rates,
        "tried": tried,
        "chain": chain,
        "jacobian": jacobian,
        "method": ("reversible-jump MCMC, Green (1995) eq. 7; hybrid "
                   "sampler with detailed balance within each move type"),
        "note": ("dimension matching n1 + m1 == n2 + m2 is enforced for "
                 "every move; j(x) is computed from the move weights "
                 "available in each model, in both directions, because "
                 "eq. 7 needs it at x' as well as x"),
    })


# --------------------------------------------------------------------------
# §4: step-function rate for a point process
# --------------------------------------------------------------------------

def step_function_loglik(y, s, h, L):
    r"""Equation 9: :math:`\sum_i \log x(y_i) - \int_0^L x(t)\,dt`.

    ``s`` are the :math:`k` interior step positions and ``h`` the
    :math:`k+1` heights, so :math:`x(t) = h_j` on
    :math:`[s_j, s_{j+1})` with :math:`s_0 = 0`, :math:`s_{k+1} = L`.
    """
    edges = [0.0] + [float(v) for v in s] + [float(L)]
    if len(h) != len(edges) - 1:
        raise ValueError("bayrjmcmc: %d heights for %d intervals"
                         % (len(h), len(edges) - 1))
    counts = [0] * len(h)
    for v in y:
        v = float(v)
        if v < 0.0 or v > float(L):
            raise ValueError("bayrjmcmc: point %g lies outside [0, %g]"
                             % (v, float(L)))
        j = 0
        while j + 1 < len(edges) - 1 and v >= edges[j + 1]:
            j += 1
        counts[j] += 1
    out = 0.0
    for j in range(len(h)):
        hj = float(h[j])
        if hj <= 0.0:
            return float("-inf")
        if counts[j]:
            out += counts[j] * math.log(hj)
        out -= hj * (edges[j + 1] - edges[j])
    return out


def changepoint_move_probabilities(lam, k_max, cap=0.9):
    r"""The §4-3 move probabilities :math:`(\eta_k, \pi_k, b_k, d_k)`.

    :math:`b_k = c \min\{1, p(k+1)/p(k)\}` and
    :math:`d_{k+1} = c \min\{1, p(k)/p(k+1)\}` with :math:`p` Poisson,
    so :math:`b_k p(k) = d_{k+1} p(k+1)` -- the condition that would
    give certain acceptance in the corresponding sampler for :math:`k`
    alone. ``c`` is taken as large as the paper allows, subject to
    :math:`b_k + d_k \leq 0.9` for every :math:`k`. Then
    :math:`d_0 = \pi_0 = 0`, :math:`b_{k_{\max}} = 0`, and
    :math:`\eta_k = \pi_k` for :math:`k \neq 0`.
    """
    lam = float(lam)
    k_max = int(k_max)
    if lam <= 0.0:
        raise ValueError("bayrjmcmc: lam must be positive")
    if k_max < 1:
        raise ValueError("bayrjmcmc: k_max must be at least 1")
    raw_b = [min(1.0, lam / (k + 1.0)) for k in range(k_max + 1)]
    raw_d = [0.0] + [min(1.0, k / lam) for k in range(1, k_max + 1)]
    raw_b[k_max] = 0.0
    worst = max(raw_b[k] + raw_d[k] for k in range(k_max + 1))
    c = cap / worst if worst > 0.0 else cap
    b = [c * v for v in raw_b]
    d = [c * v for v in raw_d]
    eta = []
    pi = []
    for k in range(k_max + 1):
        rest = 1.0 - b[k] - d[k]
        if k == 0:
            eta.append(rest)
            pi.append(0.0)
        else:
            eta.append(0.5 * rest)
            pi.append(0.5 * rest)
    return eta, pi, b, d, c


def birth_split_heights(h_j, u, s_left, s_star, s_right):
    r"""Green's birth perturbation of a step height.

    The current height is kept as the weighted geometric mean of the
    two new ones,

    .. math::

       (s^* - s_j)\log h'_j + (s_{j+1} - s^*)\log h'_{j+1}
         = (s_{j+1} - s_j)\log h_j,

    with the split fixed by :math:`h'_{j+1}/h'_j = u/(1-u)` for
    :math:`u \sim U[0, 1]`. Positivity is preserved, which is why the
    geometric rather than arithmetic mean is used.
    """
    span = float(s_right) - float(s_left)
    if span <= 0.0:
        raise ValueError("bayrjmcmc: empty interval in a birth move")
    w1 = (float(s_star) - float(s_left)) / span
    r = float(u) / (1.0 - float(u))
    lr = math.log(r)
    hj = float(h_j) * math.exp(-(1.0 - w1) * lr)
    hj1 = float(h_j) * math.exp(w1 * lr)
    return hj, hj1


def birth_log_jacobian(h_j, h_new_left, h_new_right):
    r""":math:`\log\{(h'_j + h'_{j+1})^2 / h_j\}`, the §4-3 Jacobian.

    This is :math:`|\partial(h'_j, h'_{j+1}) / \partial(h_j, u)|` for
    the map in :func:`birth_split_heights`.
    """
    return 2.0 * math.log(float(h_new_left) + float(h_new_right)) \
        - math.log(float(h_j))


def _merge_height(s_left, s_mid, s_right, h_left, h_right):
    """The death move's weighted geometric mean, reversing the birth."""
    span = float(s_right) - float(s_left)
    return math.exp(((float(s_mid) - float(s_left)) * math.log(h_left)
                     + (float(s_right) - float(s_mid)) * math.log(h_right))
                    / span)


def _log_k_prior_ratio(lam, k):
    """log p(k+1) / p(k) for the Poisson prior; the truncation cancels."""
    return math.log(lam) - math.log(k + 1.0)


def changepoint_rjmcmc(y=(), L=1.0, n_iter=40000, burn_in=4000, lam=3.0,
                       k_max=30, alpha=1.0, beta=200.0, seed=0, cap=0.9,
                       use_likelihood=True, k_init=0, thin=1,
                       keep_chain=False):
    r"""The §4 multiple change-point sampler for a step-function rate.

    Point-process data ``y`` on :math:`[0, L]` with rate :math:`x(t)` a
    step function: :math:`k \sim \mathrm{Poisson}(\lambda)` truncated at
    ``k_max``, positions the even-numbered order statistics of
    :math:`2k+1` uniforms on :math:`[0, L]` -- which spaces the steps
    out instead of letting short, barely-penalised intervals survive --
    and heights independently :math:`\Gamma(\alpha, \beta)`.

    Four moves, with the acceptance ratios printed in §4-3: height
    change (:math:`\log(h'_j/h_j) \sim U[-\tfrac12, \tfrac12]`),
    position change (:math:`s^*_j \sim U[s_{j-1}, s_{j+1}]`), birth of a
    step at :math:`s^* \sim U[0, L]`, and death of a step drawn at
    random. Birth and death are each other's reverse and satisfy
    dimension matching: the birth takes :math:`2k+1` parameters to
    :math:`2k+3`, the difference being :math:`s^*` and the :math:`u`
    that splits the height.

    ``use_likelihood=False`` drops Equation 9, leaving the prior as the
    target -- the chain must then reproduce
    :math:`\mathrm{Poisson}(\lambda)` truncated at ``k_max``, which is
    an exact statement the sampler can fail.

    The defaults are the run of §4-4.
    """
    L = float(L)
    if L <= 0.0:
        raise ValueError("bayrjmcmc: L must be positive")
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("bayrjmcmc: alpha and beta must be positive")
    n_iter = int(n_iter)
    burn_in = int(burn_in)
    if n_iter < 1:
        raise ValueError("bayrjmcmc: n_iter must be at least 1")
    if burn_in < 0 or burn_in >= n_iter:
        raise ValueError("bayrjmcmc: burn_in must be in [0, n_iter)")
    thin = int(thin)
    if thin < 1:
        raise ValueError("bayrjmcmc: thin must be at least 1")
    y = [float(v) for v in y]
    for v in y:
        if v < 0.0 or v > L:
            raise ValueError("bayrjmcmc: point %g lies outside [0, %g]"
                             % (v, L))
    eta, pi_, b, d, c = changepoint_move_probabilities(lam, k_max, cap=cap)
    k_init = int(k_init)
    if k_init < 0 or k_init > k_max:
        raise ValueError("bayrjmcmc: k_init must be in [0, k_max]")

    uni = _unif_stream(seed)
    # start from the prior mean height and evenly spaced positions
    s = [L * (i + 1.0) / (k_init + 1.0) for i in range(k_init)]
    h = [alpha / beta] * (k_init + 1)

    def loglik(s_, h_):
        return step_function_loglik(y, s_, h_, L) if use_likelihood else 0.0

    cur_ll = loglik(s, h)
    counts = [0] * (k_max + 1)
    tried = {"height": 0, "position": 0, "birth": 0, "death": 0}
    acc = {"height": 0, "position": 0, "birth": 0, "death": 0}
    chain = []
    s1_sum = 0.0
    s1_sq = 0.0
    s1_n = 0
    h_sum = 0.0
    h_n = 0

    for it in range(n_iter):
        k = len(s)
        pick = uni()
        edges = [0.0] + s + [L]

        if pick < b[k]:
            # ---- birth (§4-3) --------------------------------------
            tried["birth"] += 1
            s_star = L * uni()
            j = 0
            while j + 1 < len(edges) - 1 and s_star >= edges[j + 1]:
                j += 1
            u = uni()
            hl, hr = birth_split_heights(h[j], u, edges[j], s_star,
                                         edges[j + 1])
            s_new = s[:j] + [s_star] + s[j:]
            h_new = h[:j] + [hl, hr] + h[j + 1:]
            new_ll = loglik(s_new, h_new)

            log_prior = (
                _log_k_prior_ratio(lam, k)
                + math.log(2.0 * (k + 1.0) * (2.0 * k + 3.0)) - 2.0 * math.log(L)
                + math.log((s_star - edges[j]) * (edges[j + 1] - s_star)
                           / (edges[j + 1] - edges[j]))
                + alpha * math.log(beta) - math.lgamma(alpha)
                + (alpha - 1.0) * math.log(hl * hr / h[j])
                - beta * (hl + hr - h[j]))
            log_prop = math.log(d[k + 1]) + math.log(L) \
                - math.log(b[k]) - math.log(k + 1.0)
            log_jac = birth_log_jacobian(h[j], hl, hr)
            log_alpha = (new_ll - cur_ll) + log_prior + log_prop + log_jac
            if k + 1 <= k_max and math.log(uni()) < log_alpha:
                s, h, cur_ll = s_new, h_new, new_ll
                acc["birth"] += 1

        elif pick < b[k] + d[k]:
            # ---- death: the same ratio, relabelled and inverted -----
            tried["death"] += 1
            i = int(uni() * k)
            if i >= k:
                i = k - 1
            j = i                      # the birth's j, in the merged state
            s_new = s[:i] + s[i + 1:]
            h_merged = _merge_height(edges[i], edges[i + 1], edges[i + 2],
                                     h[i], h[i + 1])
            h_new = h[:i] + [h_merged] + h[i + 2:]
            new_ll = loglik(s_new, h_new)
            kk = k - 1                 # the count the birth would start from
            s_star = edges[i + 1]
            left = edges[i]
            right = edges[i + 2]
            log_prior = (
                _log_k_prior_ratio(lam, kk)
                + math.log(2.0 * (kk + 1.0) * (2.0 * kk + 3.0))
                - 2.0 * math.log(L)
                + math.log((s_star - left) * (right - s_star)
                           / (right - left))
                + alpha * math.log(beta) - math.lgamma(alpha)
                + (alpha - 1.0) * math.log(h[i] * h[i + 1] / h_merged)
                - beta * (h[i] + h[i + 1] - h_merged))
            log_prop = math.log(d[kk + 1]) + math.log(L) \
                - math.log(b[kk]) - math.log(kk + 1.0)
            log_jac = birth_log_jacobian(h_merged, h[i], h[i + 1])
            log_alpha = -((cur_ll - new_ll) + log_prior + log_prop + log_jac)
            if math.log(uni()) < log_alpha:
                s, h, cur_ll = s_new, h_new, new_ll
                acc["death"] += 1

        elif pick < b[k] + d[k] + eta[k]:
            # ---- height change --------------------------------------
            tried["height"] += 1
            j = int(uni() * (k + 1))
            if j > k:
                j = k
            hj = h[j] * math.exp(uni() - 0.5)
            h_new = list(h)
            h_new[j] = hj
            new_ll = loglik(s, h_new)
            log_alpha = ((new_ll - cur_ll)
                         + alpha * math.log(hj / h[j])
                         - beta * (hj - h[j]))
            if math.log(uni()) < log_alpha:
                h, cur_ll = h_new, new_ll
                acc["height"] += 1

        elif k >= 1:
            # ---- position change ------------------------------------
            tried["position"] += 1
            j = int(uni() * k)
            if j >= k:
                j = k - 1
            lo = edges[j]
            hi = edges[j + 2]
            s_star = lo + (hi - lo) * uni()
            s_new = list(s)
            s_new[j] = s_star
            new_ll = loglik(s_new, h)
            log_alpha = (new_ll - cur_ll
                         + math.log((hi - s_star) * (s_star - lo)
                                    / ((hi - s[j]) * (s[j] - lo))))
            if math.log(uni()) < log_alpha:
                s, cur_ll = s_new, new_ll
                acc["position"] += 1

        if it >= burn_in:
            counts[len(s)] += 1
            if len(s) == 1:
                s1_sum += s[0]
                s1_sq += s[0] * s[0]
                s1_n += 1
            for v in h:
                h_sum += v
                h_n += 1
            if keep_chain and ((it - burn_in) % thin == 0):
                chain.append((list(s), list(h)))

    kept = sum(counts)
    k_post = [v / float(kept) for v in counts]
    rates = dict((key, acc[key] / float(tried[key]))
                 for key in tried if tried[key])
    return RichResult(payload={
        "k_posterior": k_post,
        "k_counts": counts,
        "k_mean": sum(i * k_post[i] for i in range(len(k_post))),
        "s": list(s),
        "h": list(h),
        "accept": rates,
        "tried": tried,
        "c": c,
        "b": b,
        "d": d,
        "eta": eta,
        "pi": pi_,
        "mean_s1_given_k1": (s1_sum / s1_n) if s1_n else float("nan"),
        "var_s1_given_k1": ((s1_sq / s1_n - (s1_sum / s1_n) ** 2)
                            if s1_n > 1 else float("nan")),
        "mean_height": (h_sum / h_n) if h_n else float("nan"),
        "chain": chain,
        "n_kept": kept,
        "use_likelihood": bool(use_likelihood),
        "method": ("Green (1995) §4 change-point sampler: height, "
                   "position, birth and death moves on a step-function "
                   "rate, with the §4-3 acceptance ratios"),
        "note": ("with use_likelihood=False the target is the prior, so "
                 "k must come back Poisson(lam) truncated at k_max and "
                 "the heights Gamma(alpha, beta)"),
    })


bayrjmcmc = reversible_jump_mcmc


def cheatsheet():
    return ("bayrjmcmc: reversible-jump MCMC (Green 1995). A hybrid "
            "sampler over models of differing dimension: pad both sides "
            "with random numbers until (theta1, u1) and (theta2, u2) are "
            "in bijection -- dimension matching, n1 + m1 = n2 + m2 -- "
            "then accept by eq. 7, the posterior ratio times the "
            "move-probability ratio times the proposal-density ratio "
            "times the Jacobian of the bijection. "
            "reversible_jump_mcmc() is the general engine (jacobian= "
            "'analytic' or 'numeric'); changepoint_rjmcmc() is the "
            "paper's step-function application with its birth/death "
            "pair, and use_likelihood=False makes it sample the prior, "
            "which is exactly known.")
