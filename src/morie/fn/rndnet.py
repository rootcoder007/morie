r"""Random Network Distillation: a novelty bonus from a random target.

Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019) "Exploration by
Random Network Distillation", *ICLR*, arXiv:1810.12894.

Two networks. A **target** :math:`f : \mathcal{O} \to \mathbb{R}^k`,
randomly initialised and then frozen forever. A **predictor**
:math:`\hat f(\cdot\,; \theta)`, trained by gradient descent on the
agent's own observation stream to minimise
:math:`\|\hat f(x; \theta) - f(x)\|^2`. The exploration bonus is that
same error,

.. math:: r^i_t = \|\hat f(x_t; \theta) - f(x_t)\|^2 .

The bonus is high on observations unlike those the predictor has been
trained on, and falls as they are revisited. Section 2.2.1 is the
reason the *target* is random rather than, say, a forward dynamics
model: prediction error has four sources, and only the first (amount of
training data) is the one worth rewarding. A random, deterministic
target that lies inside the predictor's own model class removes sources
2 (stochasticity -- the "noisy-TV" failure, where an agent is drawn to a
screen of static because it can never predict it) and 3 (model
misspecification) by construction. ``anchor_rndnet.py`` runs the
noisy-TV experiment and checks this actually holds.

Everything in section 2.4 matters in practice and is implemented:

**Observation normalisation** is called *crucial* in the paper, not
optional, precisely because the target's parameters are frozen and
cannot adapt to the scale of the input. Each dimension is whitened by a
running mean and standard deviation and then clipped to
:math:`[-5, 5]`. The running statistics are initialised from a short
random-agent rollout ``init_steps`` before optimisation starts; the same
normalisation feeds predictor and target, and not the policy.

**Intrinsic-reward normalisation** divides :math:`r^i` by a running
estimate of the standard deviation of the intrinsic *returns* (not the
rewards), which is what keeps the bonus on a comparable scale across
environments and across training.

**Two value heads.** Section 2.3: the return is linear in the reward,
so :math:`R = R_E + R_I` decomposes, and the two streams are fitted
separately and added, :math:`V = V_E + V_I`. This lets the intrinsic
stream be non-episodic (not truncated at "game over") while the
extrinsic stream stays episodic, and lets the two carry different
discounts. :func:`combine_returns` does that arithmetic.

The networks here are single-hidden-layer with a fixed random target and
a predictor trained by plain SGD, which is enough for the bonus to be
exactly what the paper defines and small enough to run anywhere. Swap
in your own by passing ``target`` and ``predictor`` callables.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rndnet", "random_network_distillation", "combine_returns"]


class _RandomFeatures(object):
    """Frozen random target: x -> tanh(W1 x + b1) W2, W fixed at init."""

    def __init__(self, n_in, n_hidden, n_out, rng, scale=1.0):
        s1 = scale / math.sqrt(max(1, n_in))
        s2 = scale / math.sqrt(max(1, n_hidden))
        self.W1 = [[(rng.random() * 2.0 - 1.0) * s1 for _ in range(n_hidden)]
                   for _ in range(n_in)]
        self.b1 = [(rng.random() * 2.0 - 1.0) * s1 for _ in range(n_hidden)]
        self.W2 = [[(rng.random() * 2.0 - 1.0) * s2 for _ in range(n_out)]
                   for _ in range(n_hidden)]
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out

    def hidden(self, x):
        h = list(self.b1)
        for j in range(self.n_in):
            xj = x[j]
            if xj == 0.0:
                continue
            row = self.W1[j]
            for i in range(self.n_hidden):
                h[i] += row[i] * xj
        return [math.tanh(v) for v in h]

    def __call__(self, x):
        h = self.hidden(x)
        out = [0.0] * self.n_out
        for i in range(self.n_hidden):
            hi = h[i]
            if hi == 0.0:
                continue
            row = self.W2[i]
            for o in range(self.n_out):
                out[o] += row[o] * hi
        return out


class _Predictor(object):
    """Same random hidden layer, trainable output layer, SGD on MSE.

    Keeping the features fixed and learning the read-out is the smallest
    model that is genuinely *inside the target's model class*, which is
    the property section 2.2.1 requires for RND to remove sources 2 and
    3 of prediction error.
    """

    def __init__(self, n_in, n_hidden, n_out, rng, scale=1.0):
        self.feat = _RandomFeatures(n_in, n_hidden, n_out, rng, scale)
        self.W = [[0.0] * n_out for _ in range(n_hidden)]
        self.n_hidden, self.n_out = n_hidden, n_out

    def __call__(self, x):
        h = self.feat.hidden(x)
        out = [0.0] * self.n_out
        for i in range(self.n_hidden):
            hi = h[i]
            if hi == 0.0:
                continue
            row = self.W[i]
            for o in range(self.n_out):
                out[o] += row[o] * hi
        return out, h

    def update(self, h, err, lr):
        for i in range(self.n_hidden):
            hi = h[i]
            if hi == 0.0:
                continue
            row = self.W[i]
            for o in range(self.n_out):
                row[o] -= lr * 2.0 * err[o] * hi


class _RunningStats(object):
    """Welford mean/variance, used for both normalisers of section 2.4."""

    def __init__(self, n):
        self.n = 0
        self.mean = [0.0] * n
        self.m2 = [0.0] * n

    def update(self, x):
        self.n += 1
        for i in range(len(x)):
            d = x[i] - self.mean[i]
            self.mean[i] += d / self.n
            self.m2[i] += d * (x[i] - self.mean[i])

    def std(self, eps=1e-8):
        if self.n < 2:
            return [1.0] * len(self.mean)
        return [math.sqrt(v / (self.n - 1)) + eps for v in self.m2]


def rndnet(observations, n_hidden=64, n_out=8, lr=0.05, clip=5.0,
           normalize_obs=True, normalize_reward=True, init_steps=0,
           gamma_int=0.99, seed=0, target=None, predictor=None,
           update=True):
    r"""Compute the RND exploration bonus along an observation stream.

    Parameters
    ----------
    observations : array-like
        ``(T, d)`` observations in the order the agent saw them. The
        order matters: the predictor is trained online, so the bonus at
        step ``t`` reflects everything before ``t``.
    n_hidden, n_out : int
        Width of the random feature layer and the embedding dimension
        :math:`k`.
    lr : float
        SGD step size for the predictor.
    clip : float
        The paper's clip on normalised observations, :math:`\pm 5`.
    normalize_obs : bool
        Section 2.4's observation whitening. Called *crucial* there;
        turning it off is supported so the effect can be measured, not
        because it is a reasonable default.
    normalize_reward : bool
        Divide :math:`r^i` by the running standard deviation of the
        intrinsic *returns*.
    init_steps : int
        Number of leading observations used only to initialise the
        normalisation statistics ("stepping a random agent in the
        environment for a small number of steps before beginning
        optimisation"). Those steps get no predictor update.
    gamma_int : float
        Discount used to form the intrinsic return for the reward
        normaliser.
    seed : int
        Seed for the target and the predictor's feature layer.
    target, predictor : callable, optional
        Bring your own :math:`f` and :math:`\hat f`. ``predictor`` must
        accept an observation and return ``(output, features)``, and
        carry an ``update(features, error, lr)`` method.
    update : bool
        If False the predictor is frozen; the bonus then measures
        novelty under the current predictor without learning. Useful for
        scoring a held-out set, which is how the paper's MNIST figure is
        produced.

    Returns
    -------
    RichResult
        ``estimate`` / ``intrinsic_reward`` is the per-step bonus
        (normalised if asked). ``raw_error`` is the unnormalised
        :math:`\|\hat f - f\|^2`, ``mse`` its mean, ``returns`` the
        discounted intrinsic returns, and ``mean_first``/``mean_last``
        the bonus averaged over the first and last tenth -- the pair
        that should fall as the predictor distils the target.

    References
    ----------
    Burda et al. (2019) arXiv:1810.12894, sections 2.2, 2.3 and 2.4.
    """
    X = [[float(v) for v in row]
         for row in np.atleast_2d(np.asarray(observations, dtype=float))]
    if not X:
        raise ValueError("rndnet: observations must be non-empty")
    d = len(X[0])
    for row in X:
        if len(row) != d:
            raise ValueError("rndnet: observations must be rectangular")
    init_steps = int(init_steps)
    if init_steps < 0 or init_steps >= len(X):
        raise ValueError("rndnet: init_steps must lie in [0, T)")
    clip = float(clip)
    if not clip > 0.0:
        raise ValueError("rndnet: clip must be > 0")

    rng = np.random.default_rng(seed)
    tgt = target if target is not None else _RandomFeatures(
        d, n_hidden, n_out, rng)
    prd = predictor if predictor is not None else _Predictor(
        d, n_hidden, n_out, np.random.default_rng(seed + 1))

    obs_stats = _RunningStats(d)
    for t in range(init_steps):
        obs_stats.update(X[t])

    ret_stats = _RunningStats(1)
    running_return = 0.0
    raw = []
    rewards = []
    returns = []
    for t in range(init_steps, len(X)):
        x = X[t]
        if normalize_obs:
            obs_stats.update(x)
            sd = obs_stats.std()
            z = []
            for i in range(d):
                v = (x[i] - obs_stats.mean[i]) / sd[i]
                z.append(max(-clip, min(clip, v)))
        else:
            z = list(x)
        ft = tgt(z)
        fh, h = prd(z)
        err = [fh[o] - ft[o] for o in range(len(ft))]
        e2 = sum(v * v for v in err)
        raw.append(e2)
        running_return = gamma_int * running_return + e2
        returns.append(running_return)
        ret_stats.update([running_return])
        if normalize_reward:
            rewards.append(e2 / ret_stats.std()[0])
        else:
            rewards.append(e2)
        if update:
            prd.update(h, err, lr)

    n = len(rewards)
    tenth = max(1, n // 10)
    return RichResult(payload={
        "estimate": rewards,
        "intrinsic_reward": rewards,
        "raw_error": raw,
        "returns": returns,
        "mse": float(sum(raw) / n),
        "mean_first": float(sum(rewards[:tenth]) / tenth),
        "mean_last": float(sum(rewards[-tenth:]) / tenth),
        "n": n,
        "target": tgt,
        "predictor": prd,
        "method": "RND (Burda et al. 2019)",
    })


def combine_returns(reward_ext, reward_int, gamma_ext=0.999, gamma_int=0.99,
                    done=None):
    r"""Section 2.3: :math:`R = R_E + R_I`, fitted as two heads.

    The extrinsic stream is **episodic** -- its return is truncated at
    each ``done`` -- while the intrinsic stream is **non-episodic**, so
    it runs straight through "game over". The paper's argument is that
    an agent should value novelty it can reach later even at the cost of
    ending this episode, but must not be able to farm an extrinsic
    reward by deliberately dying in a loop.

    Returns the two return series and their sum, which is the target for
    :math:`V = V_E + V_I`.
    """
    re = [float(v) for v in np.atleast_1d(np.asarray(reward_ext,
                                                     dtype=float))]
    ri = [float(v) for v in np.atleast_1d(np.asarray(reward_int,
                                                     dtype=float))]
    if len(re) != len(ri):
        raise ValueError("combine_returns: the two reward streams must have "
                         "the same length")
    T = len(re)
    d = [False] * T if done is None else [bool(v) for v in done]
    if len(d) != T:
        raise ValueError("combine_returns: done must match the reward length")
    Re = [0.0] * T
    Ri = [0.0] * T
    acc_e = 0.0
    acc_i = 0.0
    for t in range(T - 1, -1, -1):
        acc_e = re[t] + (0.0 if d[t] else gamma_ext * acc_e)
        acc_i = ri[t] + gamma_int * acc_i          # never truncated
        Re[t] = acc_e
        Ri[t] = acc_i
    return RichResult(payload={
        "estimate": [Re[t] + Ri[t] for t in range(T)],
        "return_ext": Re,
        "return_int": Ri,
        "return_total": [Re[t] + Ri[t] for t in range(T)],
        "gamma_ext": float(gamma_ext),
        "gamma_int": float(gamma_int),
        "method": "RND two value heads (Burda et al. 2019 sec. 2.3)",
    })


def cheatsheet():
    return ("rndnet: RND bonus r^i = ||fhat(x) - f(x)||^2 with f a "
            "FROZEN random net (Burda 2019). Random deterministic "
            "target kills the noisy-TV problem. Obs whitened and "
            "clipped to +-5, r^i divided by the running std of "
            "intrinsic RETURNS (sec 2.4). combine_returns is V = V_E + "
            "V_I with the intrinsic stream non-episodic (sec 2.3).")


# compact alias per ledger/NAMING.md
random_network_distillation = rndnet
