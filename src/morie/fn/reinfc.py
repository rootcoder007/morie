r"""REINFORCE: episodic and immediate-reinforcement policy gradients.

Williams, R. J. (1992) "Simple Statistical Gradient-Following Algorithms
for Connectionist Reinforcement Learning", *Machine Learning* 8,
229-256.

A REINFORCE algorithm (the paper's eq. 2) updates every parameter by

.. math:: \Delta w_{ij} = \alpha_{ij}\,(r - b_{ij})\,
          \frac{\partial \ln g_i}{\partial w_{ij}},

where :math:`g_i` is the unit's probability mass/density function,
:math:`r` the reinforcement, :math:`b_{ij}` the *reinforcement
baseline*, and :math:`\alpha_{ij}` the rate factor. The derivative
:math:`\partial \ln g_i / \partial w_{ij}` is what Williams calls the
**characteristic eligibility** of :math:`w_{ij}`.

Theorem 1 is the reason the family is interesting: for *any* such
algorithm

.. math:: E\{\Delta W \mid W\}^{T}\, \nabla_W E\{r \mid W\} \ge 0,

i.e. the average update lies along the gradient of expected
reinforcement, with equality iff that gradient vanishes. The baseline
:math:`b` drops out of the expectation entirely -- it changes the
variance, never the mean. ``anchor_reinfc.py`` checks exactly that, in
closed form rather than by simulation.

Units implemented, each straight out of the paper:

``"bernoulli"``
    A two-action stochastic learning automaton with the action
    probability :math:`p` *itself* as the parameter. Eq. 5 gives
    :math:`\partial \ln g / \partial p = (y - p) / (p(1-p))`, and with
    the rate factor :math:`\alpha = \rho\, p (1-p)` of the paper the
    update collapses to

    .. math:: \Delta p = \rho\, r\, (y - p),

    which is the 2-action linear reward-inaction (:math:`L_{R-I}`)
    automaton.

``"bernoulli-logistic"``
    A Bernoulli semilinear unit with the logistic squashing function.
    Eq. 7 gives the characteristic eligibility :math:`(y_i - p_i) x_j`
    and hence eq. 8,

    .. math:: \Delta w_{ij} = \alpha\, r\, (y_i - p_i)\, x_j,

    or eq. 9 with reinforcement comparison,
    :math:`\Delta w_{ij} = \alpha (r - \bar r)(y_i - p_i) x_j`.

``"gaussian"``
    A Gaussian unit with adaptable mean and standard deviation. Eqs. 13
    and 14:

    .. math:: \Delta \mu = \alpha_\mu (r - b_\mu)\frac{y - \mu}{\sigma^2},
              \qquad
              \Delta \sigma = \alpha_\sigma (r - b_\sigma)
              \frac{(y-\mu)^2 - \sigma^2}{\sigma^3}.

    The paper's "reasonable algorithm" sets
    :math:`\alpha_\mu = \alpha_\sigma = \alpha \sigma^2`; that is the
    default here and is what ``rate_scaling="sigma2"`` means.

Baselines (``baseline=``):

``"none"``
    :math:`b = 0`.
``"comparison"``
    Reinforcement comparison (Sutton 1984), the paper's eq. 10:
    :math:`\bar r(t) = \gamma\, \bar r(t-1) + (1-\gamma)\, r(t-1)`.
    Note the lag -- :math:`\bar r` is never allowed to depend on the
    current :math:`r`, which is precisely the condition under which
    eq. 9 remains a REINFORCE algorithm.
``"mean"``
    The running mean of all reinforcement seen *before* the current
    trial. Same lag condition, no decay parameter to pick.

Modes (``mode=``):

``"immediate"``
    Theorem 1. One reinforcement per trial, one update per trial.
``"episodic"``
    Theorem 2. Reinforcement arrives once at the end of a
    :math:`k`-step episode and the eligibilities are *accumulated*
    across the episode before a single update is applied:

    .. math:: \Delta w_{ij} = \alpha_{ij}(r - b_{ij}) \sum_{t=1}^{k}
              e_{ij}(t).

    Williams' Lemma 3 / Theorem 2 prove this inherits the alignment
    property via the unfolding-in-time net :math:`N^*`. Section 8.1 is
    honest that it is "especially slow", because credit is spread
    uniformly over all past times; that is a property of the algorithm,
    not of this implementation.

Section 8.4's variant (replace :math:`p` by an exponentially averaged
:math:`\bar y`) is offered as ``eligibility="ybar"``; the paper reports
it converges faster but gives no derivation, so it is not the default.

Nothing here trains a neural network -- the paper's algorithms are
defined on the parameters of the stochastic units themselves, and that
is what is implemented, exactly and deterministically given a seed.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["reinfc", "reinforce"]

_UNITS = ("bernoulli", "bernoulli-logistic", "gaussian")
_BASELINES = ("none", "comparison", "mean")
_MODES = ("immediate", "episodic")


def _logistic(s):
    if s >= 0.0:
        return 1.0 / (1.0 + math.exp(-s))
    e = math.exp(s)
    return e / (1.0 + e)


def _as_matrix(x, name):
    rows = [list(map(float, r)) for r in np.atleast_2d(np.asarray(x, dtype=float))]
    if not rows or not rows[0]:
        raise ValueError("reinfc: %s must be non-empty" % name)
    width = len(rows[0])
    for r in rows:
        if len(r) != width:
            raise ValueError("reinfc: %s must be rectangular" % name)
    return rows


def _baseline_series(rewards, baseline, gamma):
    """b(t), computed only from reinforcement strictly before trial t.

    Williams eq. 10 plus the standing condition of eq. 9 that the
    baseline may never depend on the current r.
    """
    n = len(rewards)
    if baseline == "none":
        return [0.0] * n
    out = [0.0] * n
    if baseline == "comparison":
        rbar = 0.0
        for t in range(n):
            out[t] = rbar
            rbar = gamma * rbar + (1.0 - gamma) * rewards[t]
        return out
    total = 0.0
    for t in range(n):
        out[t] = total / t if t else 0.0
        total += rewards[t]
    return out


def reinfc(reward_fn, x=None, w=None, p=None, mu=0.0, sigma=1.0,
           unit="bernoulli-logistic", baseline="comparison", mode="immediate",
           alpha=0.1, gamma=0.9, rho=0.1, episode_length=1, trials=100,
           eligibility="p", rate_scaling="sigma2", seed=0):
    r"""Run a REINFORCE algorithm and return its parameter trajectory.

    Parameters
    ----------
    reward_fn : callable
        ``reward_fn(y, x)`` -> float, the environment's reinforcement.
        ``y`` is the unit's output for the trial (a list, one entry per
        unit, for the Bernoulli units; a float for ``"gaussian"``). In
        ``mode="episodic"`` it is called once per episode with the list
        of the episode's outputs.
    x : array-like, optional
        Input patterns, one row per trial, for the semilinear unit. A
        single row is reused for every trial. Defaults to a lone bias
        input of 1.0.
    w : array-like, optional
        Initial weights, shape ``(n_units, n_inputs)``. Defaults to
        zeros, i.e. :math:`p_i = 1/2`.
    p : array-like, optional
        Initial action probabilities for ``unit="bernoulli"``.
    mu, sigma : float
        Initial parameters for ``unit="gaussian"``.
    unit : {"bernoulli-logistic", "bernoulli", "gaussian"}
        Which of the paper's stochastic units to adapt.
    baseline : {"comparison", "none", "mean"}
        Reinforcement baseline; see the module docstring.
    mode : {"immediate", "episodic"}
        Theorem 1 or Theorem 2.
    alpha : float
        Learning rate :math:`\alpha`. For the Gaussian unit it is the
        :math:`\alpha` of :math:`\alpha_\mu = \alpha \sigma^2` when
        ``rate_scaling="sigma2"``.
    gamma : float
        Decay :math:`\gamma` of eq. 10, used when
        ``baseline="comparison"``.
    rho : float
        Rate factor :math:`\rho` of the :math:`L_{R-I}` form, used when
        ``unit="bernoulli"``.
    episode_length : int
        :math:`k`, the number of steps per episode when
        ``mode="episodic"``.
    trials : int
        Number of trials (episodes, when ``mode="episodic"``).
    eligibility : {"p", "ybar"}
        ``"p"`` is the paper's derived eligibility. ``"ybar"`` is the
        section 8.4 variant that substitutes an exponentially averaged
        output for the mean parameter.
    rate_scaling : {"sigma2", "none"}
        Gaussian unit only.
    seed : int
        Seed for the unit's own randomness.

    Returns
    -------
    RichResult
        ``estimate`` is the final parameter vector (``w`` flattened,
        ``p``, or ``[mu, sigma]``). Also carries ``rewards``, the
        per-trial reinforcement; ``baseline``, the :math:`b(t)` actually
        used; ``trajectory``, the parameter after every trial; and
        ``mean_reward_first``/``mean_reward_last``, the mean
        reinforcement over the first and last tenth of the run.

    References
    ----------
    Williams, R. J. (1992) *Machine Learning* 8, 229-256, eqs. 2, 5, 7,
    8, 9, 10, 13, 14 and Theorems 1-2.
    """
    if unit not in _UNITS:
        raise ValueError("reinfc: unit must be one of %r, got %r"
                         % (_UNITS, unit))
    if baseline not in _BASELINES:
        raise ValueError("reinfc: baseline must be one of %r, got %r"
                         % (_BASELINES, baseline))
    if mode not in _MODES:
        raise ValueError("reinfc: mode must be one of %r, got %r"
                         % (_MODES, mode))
    if eligibility not in ("p", "ybar"):
        raise ValueError("reinfc: eligibility must be 'p' or 'ybar', got %r"
                         % (eligibility,))
    if not callable(reward_fn):
        raise TypeError("reinfc: reward_fn must be callable")
    trials = int(trials)
    if trials < 1:
        raise ValueError("reinfc: trials must be >= 1")
    k = int(episode_length)
    if k < 1:
        raise ValueError("reinfc: episode_length must be >= 1")
    if mode == "immediate":
        k = 1

    rng = np.random.default_rng(seed)

    if unit == "gaussian":
        if float(sigma) <= 0.0:
            raise ValueError("reinfc: sigma must be > 0")
        return _run_gaussian(reward_fn, float(mu), float(sigma), baseline,
                             mode, float(alpha), float(gamma), k, trials,
                             rate_scaling, rng)

    if unit == "bernoulli":
        pv = [0.5] if p is None else [float(v) for v in np.atleast_1d(
            np.asarray(p, dtype=float))]
        for v in pv:
            if not 0.0 < v < 1.0:
                raise ValueError("reinfc: p must lie strictly in (0, 1)")
        return _run_bernoulli(reward_fn, pv, baseline, mode, float(rho),
                              float(gamma), k, trials, rng)

    xs = [[1.0]] if x is None else _as_matrix(x, "x")
    n_in = len(xs[0])
    if w is None:
        wm = [[0.0] * n_in]
    else:
        wm = _as_matrix(w, "w")
        if len(wm[0]) != n_in:
            raise ValueError("reinfc: w has %d columns but x has %d"
                             % (len(wm[0]), n_in))
    return _run_logistic(reward_fn, xs, wm, baseline, mode, float(alpha),
                         float(gamma), k, trials, eligibility, rng)


def _finish(param, rewards, bs, traj):
    n = len(rewards)
    tenth = max(1, n // 10)
    return RichResult(payload={
        "estimate": list(param),
        "rewards": list(rewards),
        "baseline": list(bs),
        "trajectory": traj,
        "n_trials": n,
        "mean_reward_first": float(sum(rewards[:tenth]) / tenth),
        "mean_reward_last": float(sum(rewards[-tenth:]) / tenth),
        "method": "REINFORCE (Williams 1992)",
    })


def _running_baseline(state, baseline, gamma):
    """Current b, given the accumulator state; b never sees the current r."""
    if baseline == "none":
        return 0.0
    if baseline == "comparison":
        return state[0]
    return state[0] / state[1] if state[1] else 0.0


def _advance_baseline(state, baseline, gamma, r):
    if baseline == "comparison":
        state[0] = gamma * state[0] + (1.0 - gamma) * r
    elif baseline == "mean":
        state[0] += r
        state[1] += 1.0


def _run_bernoulli(reward_fn, pv, baseline, mode, rho, gamma, k, trials, rng):
    r"""Eq. 5 with the rate factor alpha = rho p (1-p): Delta p = rho r (y-p).

    With ``baseline="none"`` and r in {0,1} this is exactly the 2-action
    linear reward-inaction automaton the paper identifies.
    """
    n = len(pv)
    p = list(pv)
    state = [0.0, 0.0]
    rewards = []
    bs = []
    traj = []
    for _ in range(trials):
        elig = [0.0] * n
        ys = []
        for _step in range(k):
            y = [1.0 if rng.random() < p[i] else 0.0 for i in range(n)]
            ys.append(y)
            for i in range(n):
                elig[i] += y[i] - p[i]
        r = float(reward_fn(ys if mode == "episodic" else ys[0], None))
        b = _running_baseline(state, baseline, gamma)
        for i in range(n):
            p[i] += rho * (r - b) * elig[i]
            p[i] = min(1.0 - 1e-12, max(1e-12, p[i]))
        _advance_baseline(state, baseline, gamma, r)
        rewards.append(r)
        bs.append(b)
        traj.append(list(p))
    return _finish(p, rewards, bs, traj)


def _run_logistic(reward_fn, xs, wm, baseline, mode, alpha, gamma, k, trials,
                  eligibility, rng):
    r"""Eqs. 7-9: Delta w_ij = alpha (r - b) (y_i - p_i) x_j."""
    n_units = len(wm)
    n_in = len(wm[0])
    w = [list(row) for row in wm]
    state = [0.0, 0.0]
    ybar = [0.5] * n_units
    rewards = []
    bs = []
    traj = []
    for t in range(trials):
        elig = [[0.0] * n_in for _ in range(n_units)]
        ys = []
        for step in range(k):
            xrow = xs[(t * k + step) % len(xs)]
            y = []
            for i in range(n_units):
                s = 0.0
                for j in range(n_in):
                    s += w[i][j] * xrow[j]
                pi = _logistic(s)
                yi = 1.0 if rng.random() < pi else 0.0
                y.append(yi)
                ref = ybar[i] if eligibility == "ybar" else pi
                for j in range(n_in):
                    elig[i][j] += (yi - ref) * xrow[j]
            ys.append(y)
        r = float(reward_fn(ys if mode == "episodic" else ys[0],
                            xs[(t * k) % len(xs)]))
        b = _running_baseline(state, baseline, gamma)
        for i in range(n_units):
            for j in range(n_in):
                w[i][j] += alpha * (r - b) * elig[i][j]
        if eligibility == "ybar":
            for i in range(n_units):
                mean_y = sum(y[i] for y in ys) / float(k)
                ybar[i] = gamma * ybar[i] + (1.0 - gamma) * mean_y
        _advance_baseline(state, baseline, gamma, r)
        rewards.append(r)
        bs.append(b)
        traj.append([v for row in w for v in row])
    flat = [v for row in w for v in row]
    res = _finish(flat, rewards, bs, traj)
    res["weights"] = [list(row) for row in w]
    return res


def _run_gaussian(reward_fn, mu, sigma, baseline, mode, alpha, gamma, k,
                  trials, rate_scaling, rng):
    r"""Eqs. 13-14 with the paper's alpha_mu = alpha_sigma = alpha sigma^2."""
    if rate_scaling not in ("sigma2", "none"):
        raise ValueError("reinfc: rate_scaling must be 'sigma2' or 'none'")
    state = [0.0, 0.0]
    rewards = []
    bs = []
    traj = []
    for _ in range(trials):
        e_mu = 0.0
        e_sig = 0.0
        ys = []
        for _step in range(k):
            y = mu + sigma * rng.standard_normal()
            ys.append(y)
            e_mu += (y - mu) / (sigma * sigma)
            e_sig += ((y - mu) ** 2 - sigma * sigma) / (sigma ** 3)
        r = float(reward_fn(ys if mode == "episodic" else ys[0], None))
        b = _running_baseline(state, baseline, gamma)
        rate = alpha * sigma * sigma if rate_scaling == "sigma2" else alpha
        mu += rate * (r - b) * e_mu
        sigma += rate * (r - b) * e_sig
        if sigma <= 1e-12:
            sigma = 1e-12
        _advance_baseline(state, baseline, gamma, r)
        rewards.append(r)
        bs.append(b)
        traj.append([mu, sigma])
    res = _finish([mu, sigma], rewards, bs, traj)
    res["mu"] = float(mu)
    res["sigma"] = float(sigma)
    return res


def expected_update(p, r0, r1, alpha=1.0, b=0.0):
    r"""Closed-form :math:`E\{\Delta w\}` for a bias-only Bernoulli-logistic
    unit whose reinforcement is :math:`r_0` on output 0 and :math:`r_1` on
    output 1.

    Theorem 1 says this must equal
    :math:`\alpha\, \partial E\{r \mid w\}/\partial w
    = \alpha\, p(1-p)(r_1 - r_0)`, for *every* baseline ``b``. Returned
    as a pair so a caller can compare the two directly; the anchors do.
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("expected_update: p must lie strictly in (0, 1)")
    upd = alpha * ((1.0 - p) * (r0 - b) * (0.0 - p)
                   + p * (r1 - b) * (1.0 - p))
    grad = alpha * p * (1.0 - p) * (r1 - r0)
    return float(upd), float(grad)


def cheatsheet():
    return ("reinfc: REINFORCE, Delta w = alpha (r - b) dln g/dw "
            "(Williams 1992 eq. 2). Units bernoulli (eq. 5, L_R-I), "
            "bernoulli-logistic (eqs. 7-9), gaussian (eqs. 13-14); "
            "baselines none/comparison (eq. 10)/mean; modes immediate "
            "(Thm 1) and episodic (Thm 2, eligibilities summed over the "
            "episode). E{dW}'grad E{r} >= 0 for every baseline.")


# compact alias per ledger/NAMING.md
reinforce = reinfc
