r"""Dreamer: value estimation and actor-critic learning in latent
imagination.

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020) "Dream to
Control: Learning Behaviors by Latent Imagination", *ICLR*,
arXiv:1912.01603.

Dreamer learns a latent dynamics model and then learns its *behaviour*
entirely inside that model -- it imagines trajectories forward from
model states and never touches the environment while doing so. The
latent model has three components (eq. 1):

.. math:: \text{representation } p(s_t \mid s_{t-1}, a_{t-1}, o_t),
          \qquad
          \text{transition } q(s_t \mid s_{t-1}, a_{t-1}),
          \qquad
          \text{reward } q(r_t \mid s_t).

The transition model is what makes imagination possible: it predicts
future model states *without* seeing the observations that would cause
them.

Behaviour learning happens over imagined trajectories
:math:`\{s_\tau, a_\tau, r_\tau\}_{\tau=t}^{t+H}`, and the paper gives
three value estimators trading bias against variance:

.. math:: V_R(s_\tau) = \mathbb{E}\Big[\sum_{n=\tau}^{t+H} r_n\Big],
          \tag{4}

which simply sums rewards to the horizon and ignores everything beyond
it -- so it needs no value model at all, and the paper uses it as an
ablation;

.. math:: V_N^k(s_\tau) = \mathbb{E}\Big[
          \sum_{n=\tau}^{h-1} \gamma^{n-\tau} r_n
          + \gamma^{h-\tau} v_\psi(s_h)\Big],
          \quad h = \min(\tau + k,\ t + H),
          \tag{5}

the :math:`k`-step estimate bootstrapping from the learned value; and
the one Dreamer actually uses,

.. math:: V_\lambda(s_\tau) = (1-\lambda)
          \sum_{n=1}^{H-1} \lambda^{n-1} V_N^n(s_\tau)
          + \lambda^{H-1} V_N^H(s_\tau),
          \tag{6}

an exponentially weighted average over horizons. Note the
:math:`\min` in eq. 5: past the imagination horizon the estimate stops
extending, so every :math:`V_N^k` with :math:`\tau + k \ge t+H`
collapses to the same value -- which is exactly why the last term of
eq. 6 carries the remaining weight :math:`\lambda^{H-1}` rather than
continuing the sum.

The updates (Algorithm 1) are then

.. math:: \phi \leftarrow \phi + \alpha \nabla_\phi
          \sum_{\tau=t}^{t+H} V_\lambda(s_\tau),
          \qquad
          \psi \leftarrow \psi - \alpha \nabla_\psi
          \sum_{\tau=t}^{t+H} \tfrac12
          \big\|v_\psi(s_\tau) - V_\lambda(s_\tau)\big\|^2 .

The action model is updated by *propagating gradients of the value
estimates back through the learned dynamics* -- that is the analytic
gradient Dreamer gets and a model-free method does not, and it is why
the paper can solve long-horizon tasks robustly with respect to
:math:`H`.

Implemented here: the three estimators of eqs. 4-6 on an imagined
trajectory, :func:`imagine` to roll one out through a supplied
transition and reward model, and :func:`value_update` for the critic's
regression target. Bring your own :math:`q(s' \mid s,a)`,
:math:`q(r \mid s)` and :math:`v_\psi`, learned or exact -- exact ones
make the estimators checkable against closed forms, which is what the
anchors do.
"""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["dreamr", "dreamer", "lambda_return", "imagine",
           "value_update"]


def _vec(x, name):
    v = [float(t) for t in np.atleast_1d(np.asarray(x, dtype=float))]
    if not v:
        raise ValueError("dreamr: %s must be non-empty" % name)
    return v


def imagine(state, action_model, transition, reward_model, horizon,
            value_model=None):
    r"""Roll an imagined trajectory forward for ``horizon`` steps.

    ``transition(s, a) -> s'`` is the paper's :math:`q(s_t \mid
    s_{t-1}, a_{t-1})` and never sees an observation. ``reward_model(s)
    -> r`` is :math:`q(r_t \mid s_t)`. Returns the states, actions,
    rewards and (if a value model is given) values along the imagined
    trajectory.
    """
    H = int(horizon)
    if H < 1:
        raise ValueError("dreamr: horizon must be >= 1")
    for fn, name in ((action_model, "action_model"),
                     (transition, "transition"),
                     (reward_model, "reward_model")):
        if not callable(fn):
            raise TypeError("dreamr: %s must be callable" % name)
    states = [state]
    actions = []
    rewards = []
    s = state
    for _ in range(H):
        a = action_model(s)
        r = float(reward_model(s))
        actions.append(a)
        rewards.append(r)
        s = transition(s, a)
        states.append(s)
    values = ([float(value_model(x)) for x in states]
              if value_model is not None else None)
    return RichResult(payload={
        "estimate": rewards,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "horizon": H,
        "method": "Dreamer latent imagination (Hafner et al. 2020 eq. 1)",
    })


def lambda_return(rewards, values, gamma=0.99, lam=0.95, estimator="lambda",
                  k=1):
    r"""The value estimators of eqs. 4-6 at every step of an imagined
    trajectory.

    Parameters
    ----------
    rewards : array-like
        :math:`r_t, \dots, r_{t+H-1}`, the imagined rewards.
    values : array-like
        :math:`v_\psi(s_t), \dots, v_\psi(s_{t+H})` -- one more entry
        than ``rewards``, since eq. 5 can bootstrap from the state
        *after* the last reward.
    gamma : float
        Discount :math:`\gamma`.
    lam : float
        :math:`\lambda` of eq. 6.
    estimator : {"lambda", "k-step", "reward"}
        Eq. 6, eq. 5 (with the given ``k``), or eq. 4.
    k : int
        The :math:`k` of eq. 5.

    Returns
    -------
    RichResult
        ``estimate`` / ``returns`` is the estimate at each
        :math:`\tau`, one per reward.

    References
    ----------
    Hafner, Lillicrap, Ba & Norouzi (2020) arXiv:1912.01603, eqs. 4-6.
    """
    if estimator not in ("lambda", "k-step", "reward"):
        raise ValueError("dreamr: estimator must be 'lambda', 'k-step' "
                         "or 'reward', got %r" % (estimator,))
    r = _vec(rewards, "rewards")
    H = len(r)
    v = _vec(values, "values")
    if len(v) != H + 1:
        raise ValueError("dreamr: values must have one more entry than "
                         "rewards (got %d and %d)" % (len(v), H))
    gamma = float(gamma)
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("dreamr: lam must lie in [0, 1]")

    if estimator == "reward":
        # eq. 4: undiscounted sum to the horizon, no value model.
        out = []
        for tau in range(H):
            out.append(sum(r[n] for n in range(tau, H)))
        return _pack(out, "V_R (eq. 4)")

    def vn(tau, kk):
        """eq. 5, with h = min(tau + k, t + H)."""
        h = min(tau + int(kk), H)
        tot = 0.0
        for n in range(tau, h):
            tot += (gamma ** (n - tau)) * r[n]
        return tot + (gamma ** (h - tau)) * v[h]

    if estimator == "k-step":
        if int(k) < 1:
            raise ValueError("dreamr: k must be >= 1")
        return _pack([vn(tau, k) for tau in range(H)], "V_N^k (eq. 5)")

    # eq. 6
    out = []
    for tau in range(H):
        acc = 0.0
        for n in range(1, H):
            acc += (1.0 - lam) * (lam ** (n - 1)) * vn(tau, n)
        acc += (lam ** (H - 1)) * vn(tau, H)
        out.append(acc)
    return _pack(out, "V_lambda (eq. 6)")


def _pack(vals, name):
    return RichResult(payload={
        "estimate": vals,
        "returns": vals,
        "n": len(vals),
        "method": "Dreamer %s (Hafner et al. 2020)" % name,
    })


def value_update(values, targets):
    r"""The critic's loss and gradient,
    :math:`\tfrac12 \|v_\psi(s_\tau) - V_\lambda(s_\tau)\|^2` summed
    over the imagined trajectory (Algorithm 1).

    Returns the loss and the per-step residual
    :math:`v_\psi - V_\lambda`, which is :math:`\partial L /
    \partial v_\psi`.
    """
    v = _vec(values, "values")
    t = _vec(targets, "targets")
    if len(v) != len(t):
        raise ValueError("dreamr: values and targets must be the same "
                         "length")
    resid = [v[i] - t[i] for i in range(len(v))]
    loss = sum(0.5 * e * e for e in resid)
    return RichResult(payload={
        "estimate": float(loss),
        "loss": float(loss),
        "residual": resid,
        "grad": resid,
        "method": "Dreamer value loss (Hafner et al. 2020, Alg. 1)",
    })


def dreamr(state, action_model, transition, reward_model, value_model,
           horizon=15, gamma=0.99, lam=0.95, estimator="lambda", k=1):
    r"""Imagine forward from a latent state and return the value
    estimates and the critic's update.

    This is one behaviour-learning step of Algorithm 1 with the model
    held fixed: imagine :math:`H` steps, predict rewards and values,
    compute :math:`V_\lambda` by eq. 6, and form the critic's
    regression loss. The actor update is
    :math:`\nabla_\phi \sum_\tau V_\lambda(s_\tau)`, which requires
    differentiating through your own dynamics and so belongs to the
    autodiff framework holding them; ``objective`` is the quantity to
    ascend.

    Returns
    -------
    RichResult
        ``estimate`` / ``returns`` is :math:`V_\lambda` per step,
        ``objective`` their sum (the actor's ascent target),
        ``value_loss`` the critic's loss, ``residual`` its gradient
        with respect to :math:`v_\psi`, and the imagined
        ``states`` / ``actions`` / ``rewards`` / ``values``.
    """
    traj = imagine(state, action_model, transition, reward_model, horizon,
                   value_model=value_model)
    ret = lambda_return(traj["rewards"], traj["values"], gamma=gamma,
                        lam=lam, estimator=estimator, k=k)
    upd = value_update(traj["values"][:-1], ret["returns"])
    return RichResult(payload={
        "estimate": ret["returns"],
        "returns": ret["returns"],
        "objective": float(sum(ret["returns"])),
        "value_loss": upd["loss"],
        "residual": upd["residual"],
        "states": traj["states"],
        "actions": traj["actions"],
        "rewards": traj["rewards"],
        "values": traj["values"],
        "horizon": int(horizon),
        "gamma": float(gamma),
        "lam": float(lam),
        "estimator": estimator,
        "method": "Dreamer behaviour step (Hafner et al. 2020, Alg. 1)",
    })


def cheatsheet():
    return ("dreamr: learn behaviour inside a latent world model "
            "(Hafner 2020). Imagine H steps with the TRANSITION model "
            "(no observations), then V_R (eq. 4, no value model), "
            "V_N^k (eq. 5, h = min(tau+k, t+H)) or V_lambda (eq. 6, "
            "the exponentially weighted average Dreamer uses). Actor "
            "ascends sum_tau V_lambda through the dynamics; critic "
            "regresses v_psi onto V_lambda.")


# compact alias per ledger/NAMING.md
dreamer = dreamr
