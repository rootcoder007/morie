r"""Intrinsic Curiosity Module: curiosity as forward-model error in a
learned, action-relevant feature space.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017)
"Curiosity-driven Exploration by Self-supervised Prediction", *ICML*,
arXiv:1705.05363.

The ICM has three pieces. A feature map :math:`\phi`, an **inverse
dynamics** model that predicts the action from a pair of consecutive
features,

.. math:: \hat a_t = g(\phi(s_t), \phi(s_{t+1}); \theta_I),
          \qquad \min_{\theta_I} L_I(\hat a_t, a_t), \tag{2, 3}

and a **forward dynamics** model that predicts the next feature from the
current feature and the action,

.. math:: \hat\phi(s_{t+1}) = f(\phi(s_t), a_t; \theta_F),
          \qquad
          L_F = \tfrac12 \|\hat\phi(s_{t+1}) - \phi(s_{t+1})\|_2^2 .
          \tag{4, 5}

The intrinsic reward is that forward error,

.. math:: r^i_t = \frac{\eta}{2}
          \|\hat\phi(s_{t+1}) - \phi(s_{t+1})\|_2^2 , \tag{6}

and everything is trained jointly with the policy,

.. math:: \min_{\theta_P, \theta_I, \theta_F}\ -\lambda\,
          \mathbb{E}_{\pi}\big[\textstyle\sum_t r_t\big]
          + (1 - \beta) L_I + \beta L_F , \qquad
          0 \le \beta \le 1,\ \lambda > 0. \tag{7}

The inverse model is the whole trick, and it is worth being explicit
about why, because it is the difference between ICM and plain forward
dynamics. :math:`\phi` is trained *only* to support predicting the
agent's own action. It therefore has no reason to encode anything the
agent cannot influence: a swaying tree, a flickering light, a screen of
static. Those get squeezed out of the feature space, so the forward
model is never asked to predict them and the agent earns nothing for
staring at them. Predicting raw observations, by contrast, pays forever
for unpredictable nuisance. ``anchor_explor.py`` sets up exactly that
comparison and measures it rather than taking the paper's word for it.

Implemented here on tabular/continuous transitions with linear models
trained by SGD, which is enough to be the objective the paper defines
and to reproduce its qualitative claims deterministically. Discrete
actions use a softmax inverse model (the paper's "output of g is a
soft-max distribution across all possible actions and minimizing
:math:`L_I` amounts to maximum likelihood estimation"); continuous
actions use a squared-error inverse model.

Two feature routes, both from the paper:

``features="inverse"``
    :math:`\phi` is learned through the inverse model, as in eq. 2-3.
    This is ICM.
``features="identity"``
    :math:`\phi(s) = s`. Forward dynamics on raw observations -- the
    baseline ICM is arguing against, kept so the noisy-TV claim is
    testable and so users can see the difference on their own data.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["explor", "intrinsic_motivation", "icm"]

_FEATURES = ("inverse", "identity")


def _mat(x, name):
    rows = [[float(v) for v in row]
            for row in np.atleast_2d(np.asarray(x, dtype=float))]
    if not rows or not rows[0]:
        raise ValueError("explor: %s must be non-empty" % name)
    w = len(rows[0])
    for r in rows:
        if len(r) != w:
            raise ValueError("explor: %s must be rectangular" % name)
    return rows


def _matvec(W, x):
    """W is (n_in, n_out) as a list of rows."""
    n_out = len(W[0])
    out = [0.0] * n_out
    for j in range(len(x)):
        xj = x[j]
        if xj == 0.0:
            continue
        row = W[j]
        for o in range(n_out):
            out[o] += row[o] * xj
    return out


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def explor(states, actions, next_states, n_actions=None, n_features=8,
           eta=1.0, beta=0.2, lr=0.05, epochs=1, features="inverse",
           discrete=True, seed=0):
    r"""Train an ICM on a batch of transitions and return the curiosity
    reward for each.

    Parameters
    ----------
    states, next_states : array-like
        ``(T, d)`` observations :math:`s_t` and :math:`s_{t+1}`.
    actions : array-like
        ``(T,)`` action indices when ``discrete``, else ``(T, m)``
        continuous action vectors.
    n_actions : int, optional
        Size of the discrete action set; inferred from ``actions`` if
        omitted.
    n_features : int
        Dimension of :math:`\phi`. Ignored when
        ``features="identity"``.
    eta : float
        :math:`\eta > 0`, the scaling of eq. 6.
    beta : float
        :math:`\beta \in [0, 1]` of eq. 7, weighing :math:`L_F` against
        :math:`L_I`. The paper uses 0.2.
    lr : float
        SGD step size.
    epochs : int
        Passes over the batch. The rewards returned are the ones seen
        on the *last* pass, i.e. under the trained models.
    features : {"inverse", "identity"}
        Learn :math:`\phi` through the inverse model (ICM) or use the
        raw observation (plain forward dynamics).
    discrete : bool
        Whether ``actions`` are indices into a softmax or continuous
        vectors.
    seed : int
        Seed for the feature-map initialisation.

    Returns
    -------
    RichResult
        ``estimate`` / ``intrinsic_reward`` is :math:`r^i_t` per
        transition (eq. 6). Also ``forward_loss`` (:math:`L_F`, eq. 5),
        ``inverse_loss`` (:math:`L_I`, eq. 3), ``objective`` (the
        :math:`(1-\beta)L_I + \beta L_F` part of eq. 7),
        ``inverse_accuracy`` for discrete actions, ``phi`` and
        ``phi_next``, and ``loss_curve`` per epoch.

    References
    ----------
    Pathak et al. (2017) arXiv:1705.05363, eqs. 2-7.
    """
    if features not in _FEATURES:
        raise ValueError("explor: features must be one of %r, got %r"
                         % (_FEATURES, features))
    eta = float(eta)
    if not eta > 0.0:
        raise ValueError("explor: eta must be > 0")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("explor: beta must lie in [0, 1]")
    S = _mat(states, "states")
    S1 = _mat(next_states, "next_states")
    if len(S) != len(S1):
        raise ValueError("explor: states and next_states must have the "
                         "same length")
    if len(S[0]) != len(S1[0]):
        raise ValueError("explor: states and next_states must have the "
                         "same width")
    T = len(S)
    d = len(S[0])

    if discrete:
        A = [int(a) for a in np.atleast_1d(np.asarray(actions))]
        if len(A) != T:
            raise ValueError("explor: got %d actions for %d transitions"
                             % (len(A), T))
        nA = int(n_actions) if n_actions is not None else max(A) + 1
        if nA < 2:
            raise ValueError("explor: need at least 2 discrete actions")
        if min(A) < 0 or max(A) >= nA:
            raise ValueError("explor: action index out of range")
        a_dim = nA
    else:
        Ac = _mat(actions, "actions")
        if len(Ac) != T:
            raise ValueError("explor: got %d actions for %d transitions"
                             % (len(Ac), T))
        a_dim = len(Ac[0])

    rng = np.random.default_rng(seed)
    if features == "identity":
        k = d
        Wphi = None
    else:
        k = int(n_features)
        if k < 1:
            raise ValueError("explor: n_features must be >= 1")
        # Small init: tanh saturated at initialisation has no gradient,
        # and phi is trained, so it must start in its linear regime.
        s = 0.1 / math.sqrt(d)
        Wphi = [[(rng.random() * 2.0 - 1.0) * s for _ in range(k)]
                for _ in range(d)]

    def phi(x):
        if Wphi is None:
            return list(x)
        return [math.tanh(v) for v in _matvec(Wphi, x)]

    # Inverse model g: (phi(s), phi(s')) -> action.  Forward model f:
    # (phi(s), a) -> phi(s').  Both linear in their inputs.
    Winv = [[0.0] * a_dim for _ in range(2 * k)]
    Wfwd = [[0.0] * k for _ in range(k + a_dim)]

    curve = []
    for _ep in range(max(1, int(epochs))):
        rewards = []
        lf_tot = 0.0
        li_tot = 0.0
        n_correct = 0
        for t in range(T):
            p = phi(S[t])
            p1 = phi(S1[t])
            if discrete:
                avec = [0.0] * a_dim
                avec[A[t]] = 1.0
            else:
                avec = Ac[t]

            # --- inverse model, eqs. 2-3
            inp_i = p + p1
            zi = _matvec(Winv, inp_i)
            if discrete:
                pr = _softmax(zi)
                li = -math.log(max(pr[A[t]], 1e-300))
                gi = [pr[o] - avec[o] for o in range(a_dim)]
                if max(range(a_dim), key=lambda o: pr[o]) == A[t]:
                    n_correct += 1
            else:
                li = 0.5 * sum((zi[o] - avec[o]) ** 2 for o in range(a_dim))
                gi = [zi[o] - avec[o] for o in range(a_dim)]
            li_tot += li

            # --- forward model, eqs. 4-5
            inp_f = p + avec
            ph = _matvec(Wfwd, inp_f)
            ef = [ph[o] - p1[o] for o in range(k)]
            lf = 0.5 * sum(v * v for v in ef)
            lf_tot += lf
            rewards.append(eta * lf)      # eq. 6: (eta/2)||.||^2 == eta*L_F

            # --- SGD on (1-beta) L_I + beta L_F  (the eq. 7 terms that
            #     do not involve the policy).
            #
            # phi's parameters belong to theta_I: the feature encoder is
            # trained through the INVERSE loss only. That is not an
            # implementation shortcut, it is the mechanism -- phi is
            # never asked to reconstruct anything, only to support
            # predicting the agent's own action, so dimensions the agent
            # cannot influence carry no gradient and fall out of the
            # representation. Letting L_F train phi as well would give it
            # an incentive to collapse phi to a constant, which drives
            # the curiosity reward to zero while learning nothing.
            if Wphi is not None:
                dphi = [0.0] * (2 * k)
                for j in range(2 * k):
                    acc = 0.0
                    row = Winv[j]
                    for o in range(a_dim):
                        acc += row[o] * gi[o]
                    dphi[j] = acc
                for half, xin in ((0, S[t]), (1, S1[t])):
                    ph_ = p if half == 0 else p1
                    for j in range(k):
                        g = dphi[half * k + j] * (1.0 - ph_[j] * ph_[j])
                        if g == 0.0:
                            continue
                        step = lr * (1.0 - beta) * g
                        for dd in range(d):
                            if xin[dd] != 0.0:
                                Wphi[dd][j] -= step * xin[dd]
            for j in range(2 * k):
                xj = inp_i[j]
                if xj == 0.0:
                    continue
                row = Winv[j]
                for o in range(a_dim):
                    row[o] -= lr * (1.0 - beta) * gi[o] * xj
            for j in range(k + a_dim):
                xj = inp_f[j]
                if xj == 0.0:
                    continue
                row = Wfwd[j]
                for o in range(k):
                    row[o] -= lr * beta * ef[o] * xj
        curve.append(((1.0 - beta) * li_tot + beta * lf_tot) / T)

    n = len(rewards)
    tenth = max(1, n // 10)
    payload = {
        "estimate": rewards,
        "intrinsic_reward": rewards,
        "forward_loss": float(lf_tot / T),
        "inverse_loss": float(li_tot / T),
        "objective": float(curve[-1]),
        "loss_curve": curve,
        "phi": [phi(s) for s in S],
        "phi_next": [phi(s) for s in S1],
        "mean_first": float(sum(rewards[:tenth]) / tenth),
        "mean_last": float(sum(rewards[-tenth:]) / tenth),
        "eta": eta,
        "beta": beta,
        "n": n,
        "features": features,
        "method": "ICM (Pathak et al. 2017, eqs. 2-7)",
    }
    if discrete:
        payload["inverse_accuracy"] = float(n_correct) / T
    return RichResult(payload=payload)


def cheatsheet():
    return ("explor: ICM (Pathak 2017). phi learned via the INVERSE "
            "model (eqs. 2-3) so it encodes only what the agent can "
            "affect; forward model f(phi(s),a) (eq. 4); curiosity "
            "r^i = (eta/2)||phihat(s') - phi(s')||^2 (eq. 6); joint "
            "loss (1-beta)L_I + beta L_F (eq. 7). features='identity' "
            "is the raw-observation baseline that the noisy TV fools.")


# compact aliases per ledger/NAMING.md
intrinsic_motivation = explor
icm = explor
