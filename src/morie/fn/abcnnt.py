# morie.fn -- function file (rootcoder007/morie)
r"""Sequential Neural Likelihood: likelihood-free inference by flow.

Papamakarios, G., Sterratt, D. C., & Murray, I. (2019) "Sequential
Neural Likelihood: Fast Likelihood-free Inference with Autoregressive
Flows", *AISTATS 22*, PMLR 89:837-848. arXiv:1805.07226

Papamakarios, G., Pavlakou, T., & Murray, I. (2017) "Masked
Autoregressive Flow for Density Estimation", *NIPS 30*.

The simulator can be run but its likelihood cannot be written down. ABC
handles that by rejecting simulations that land far from the observation
and pays for it in wasted simulations. SNL instead *learns* the
likelihood: train a conditional density estimator
:math:`q_\phi(x \mid \theta)` on pairs the simulator produces, then treat
:math:`q_\phi(x_o \mid \theta)\,p(\theta)` as the posterior.

**Algorithm 1**, exactly as printed:

    set :math:`\hat{p}_0(\theta \mid x_o) = p(\theta)` and
    :math:`D = \{\}`; for each of :math:`R` rounds, draw :math:`N`
    parameters :math:`\theta_n \sim \hat{p}_{r-1}(\theta \mid x_o)` by
    MCMC, simulate :math:`x_n \sim p(x \mid \theta_n)`, add the pairs to
    :math:`D`, retrain :math:`q_\phi` on **all** of :math:`D`, and set
    :math:`\hat{p}_r(\theta \mid x_o) \propto q_\phi(x_o \mid \theta)
    p(\theta)`.

Two details in that loop carry the method. Training is on the whole of
:math:`D`, not the newest round, so nothing is thrown away. And the
proposal is the *current posterior estimate*, which concentrates
simulations where the observation actually is -- Equation 1 shows why
that is the right target: maximising the total log likelihood is
maximising :math:`-E_{\tilde{p}(\theta)}
[D_{KL}(p(x \mid \theta) \| q_\phi(x \mid \theta))]` up to a constant, so
:math:`q_\phi` is fitted hardest where the proposal puts mass.

**The estimator is a Masked Autoregressive Flow**, as the paper uses
("a MAF with 5 autoregressive layers"). Each layer is an affine
autoregressive transform whose shift and log-scale for dimension
:math:`i` are functions of :math:`x_{<i}` and the conditioning
:math:`\theta` only:

.. math::

   u_i = (x_i - \mu_i(x_{<i}, \theta))\,
         \exp\{-\alpha_i(x_{<i}, \theta)\},
   \qquad
   \log|\det J| = -\sum_i \alpha_i ,

so the density follows from the change of variables,
:math:`\log q_\phi(x \mid \theta) = \log N(u; 0, I) - \sum_i \alpha_i`.
The autoregressive structure is what makes the Jacobian triangular and
the determinant a sum; it is enforced by masking, and the anchor checks
it by finite differences rather than trusting the masks. Layers reverse
the variable order between them, as MAF does, so no dimension is
permanently last.

Everything runs in plain Python, so the sizes are anchor-sized rather
than the paper's.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "abcnnt",
    "sequential_neural_likelihood",
    "MAF",
    "made_layer",
    "flow_logprob",
    "flow_forward",
    "train_flow",
    "mcmc_sample",
]


def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def uni():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    def normal():
        return math.sqrt(-2.0 * math.log(max(uni(), 1e-12))) * \
            math.cos(2 * math.pi * uni())
    return uni, normal


def made_layer(dim_x, dim_t, hidden, normal, reverse=False):
    r"""One masked autoregressive layer.

    ``W1`` maps ``[x, theta]`` to hidden units and ``W2`` maps those to
    the shift and log-scale of each dimension. The mask on ``W1`` gives
    hidden unit :math:`h` a degree, and the mask on ``W2`` lets output
    :math:`i` see only hidden units of strictly smaller degree -- which
    is exactly the condition that output :math:`i` depends on
    :math:`x_{<i}` alone. The conditioning inputs are never masked,
    since :math:`\theta` may inform every dimension.
    """
    order = list(range(dim_x))
    if reverse:
        order.reverse()
    # MADE degrees, 1-based for inputs and outputs. Hidden degrees run
    # 0..dim_x-1: a degree-0 unit sees no x at all, only theta, and every
    # output may see it -- which is the path the conditioning needs. With
    # degrees starting at 1 the first output would see nothing, and with
    # dim_x = 1 no output would see anything, leaving mu and alpha as
    # constants that theta could not move.
    deg_in = [order[i] + 1 for i in range(dim_x)]
    deg_out = [order[i] + 1 for i in range(dim_x)]
    deg_h = [i % dim_x for i in range(hidden)]
    s1 = 1.0 / math.sqrt(dim_x + dim_t)
    s2 = 1.0 / math.sqrt(hidden)
    W1 = [[normal() * s1 for _ in range(dim_x + dim_t)]
          for _ in range(hidden)]
    b1 = [0.0] * hidden
    Wm = [[normal() * s2 for _ in range(hidden)] for _ in range(dim_x)]
    Wa = [[normal() * s2 for _ in range(hidden)] for _ in range(dim_x)]
    bm = [0.0] * dim_x
    ba = [0.0] * dim_x
    # M1[h][j] = 1 if deg_h[h] >= deg_in[j]; theta columns always 1
    M1 = [[1.0 if deg_h[h] >= deg_in[j] else 0.0
           for j in range(dim_x)] + [1.0] * dim_t for h in range(hidden)]
    # M2[i][h] = 1 if deg_out[i] > deg_h[h]
    M2 = [[1.0 if deg_out[i] > deg_h[h] else 0.0 for h in range(hidden)]
          for i in range(dim_x)]
    return {"W1": W1, "b1": b1, "Wm": Wm, "bm": bm, "Wa": Wa, "ba": ba,
            "M1": M1, "M2": M2, "dim_x": dim_x, "dim_t": dim_t,
            "hidden": hidden, "order": order}


def _layer_stats(layer, x, t):
    """The shift and log-scale of each dimension, given ``x`` and ``t``."""
    dx, dt, H = layer["dim_x"], layer["dim_t"], layer["hidden"]
    inp = list(x) + list(t)
    h = []
    for k in range(H):
        z = layer["b1"][k] + sum(layer["W1"][k][j] * layer["M1"][k][j] *
                                 inp[j] for j in range(dx + dt))
        h.append(math.tanh(z))
    mu, al = [], []
    for i in range(dx):
        mu.append(layer["bm"][i] +
                  sum(layer["Wm"][i][k] * layer["M2"][i][k] * h[k]
                      for k in range(H)))
        a = layer["ba"][i] + sum(layer["Wa"][i][k] * layer["M2"][i][k] *
                                 h[k] for k in range(H))
        al.append(max(min(a, 5.0), -5.0))     # keep the scale sane
    return mu, al, h


def flow_forward(flow, x, t):
    r"""Push ``x`` through every layer; returns ``(u, sum_alpha)``."""
    u = [float(v) for v in x]
    total = 0.0
    for layer in flow["layers"]:
        mu, al, _h = _layer_stats(layer, u, t)
        u = [(u[i] - mu[i]) * math.exp(-al[i])
             for i in range(layer["dim_x"])]
        total += sum(al)
    return u, total


def flow_logprob(flow, x, t):
    r"""Change of variables: :math:`\log N(u; 0, I) - \sum_i \alpha_i`."""
    u, total = flow_forward(flow, x, t)
    d = len(u)
    return (-0.5 * sum(v * v for v in u) - 0.5 * d * math.log(2 * math.pi)
            - total)


def MAF(dim_x, dim_t, n_layers=5, hidden=20, seed=0):
    """A Masked Autoregressive Flow, the paper's 5 layers by default."""
    if dim_x < 1 or dim_t < 1:
        raise ValueError("abcnnt: dimensions must be positive")
    if n_layers < 1 or hidden < 1:
        raise ValueError("abcnnt: n_layers and hidden must be positive")
    _uni, normal = _rng(seed + 17)
    return {"layers": [made_layer(dim_x, dim_t, hidden, normal,
                                  reverse=(k % 2 == 1))
                       for k in range(int(n_layers))],
            "dim_x": dim_x, "dim_t": dim_t}


def _params(flow):
    """Every trainable array, as (container, index) pairs."""
    out = []
    for L in flow["layers"]:
        for k in range(L["hidden"]):
            for j in range(L["dim_x"] + L["dim_t"]):
                if L["M1"][k][j]:
                    out.append((L["W1"][k], j))
            out.append((L["b1"], k))
        for i in range(L["dim_x"]):
            for k in range(L["hidden"]):
                if L["M2"][i][k]:
                    out.append((L["Wm"][i], k))
                    out.append((L["Wa"][i], k))
            out.append((L["bm"], i))
            out.append((L["ba"], i))
    return out


def train_flow(flow, D, epochs=40, lr=0.01, seed=0, batch=None):
    """Maximise the total log likelihood over ``D`` by SGD.

    Gradients are taken by central differences on the unmasked
    parameters. That is slow and entirely honest: the point of this
    module is Algorithm 1, and a hand-rolled backward pass would be one
    more thing to get silently wrong.
    """
    if not D:
        raise ValueError("abcnnt: no training pairs")
    if epochs < 1 or lr <= 0:
        raise ValueError("abcnnt: epochs must be >= 1 and lr positive")
    uni, _n = _rng(seed + 23)
    ps = _params(flow)
    n = len(D)
    bs = n if batch is None else max(1, min(int(batch), n))

    def total(sample):
        return sum(flow_logprob(flow, x, t) for t, x in sample) / \
            float(len(sample))

    h = 1e-4
    for _ in range(int(epochs)):
        idx = [int(uni() * n) for _ in range(bs)]
        sample = [D[i] for i in idx]
        for arr, j in ps:
            old = arr[j]
            arr[j] = old + h
            up = total(sample)
            arr[j] = old - h
            dn = total(sample)
            arr[j] = old + lr * (up - dn) / (2 * h)
    return flow


def mcmc_sample(logpdf, x0, n, burn=100, step=0.5, seed=0):
    """Random-walk Metropolis, used to draw from the posterior estimate."""
    if n < 1:
        raise ValueError("abcnnt: n must be positive")
    if step <= 0:
        raise ValueError("abcnnt: step must be positive")
    uni, normal = _rng(seed + 31)
    cur = [float(v) for v in x0]
    lc = logpdf(cur)
    out, acc = [], 0
    for it in range(int(burn) + int(n)):
        prop = [cur[i] + step * normal() for i in range(len(cur))]
        lp = logpdf(prop)
        if math.log(max(uni(), 1e-300)) < lp - lc:
            cur, lc = prop, lp
            acc += 1
        if it >= burn:
            out.append(list(cur))
    return out, acc / float(burn + n)


def abcnnt(simulator, x_o, log_prior, theta0, n_rounds=3, n_per_round=50,
           n_layers=5, hidden=20, epochs=40, lr=0.01, mcmc_burn=100,
           mcmc_step=0.5, seed=0, n_posterior=200):
    """Algorithm 1: Sequential Neural Likelihood.

    ``simulator(theta, rnd)`` returns one simulated ``x``; ``log_prior``
    is the log prior density; ``theta0`` starts the MCMC chains.
    """
    x_o = [float(v) for v in x_o]
    theta0 = [float(v) for v in theta0]
    if not x_o or not theta0:
        raise ValueError("abcnnt: x_o and theta0 must be non-empty")
    if n_rounds < 1 or n_per_round < 1:
        raise ValueError("abcnnt: n_rounds and n_per_round must be "
                         "positive")
    uni, normal = _rng(seed + 5)
    flow = MAF(len(x_o), len(theta0), n_layers, hidden, seed)
    D = []
    history = []
    # round 0's proposal is the prior itself, exactly as Algorithm 1 sets
    # p_hat_0(theta | x_o) = p(theta)
    logpost = log_prior
    for r in range(int(n_rounds)):
        draws, rate = mcmc_sample(logpost, theta0, int(n_per_round),
                                  mcmc_burn, mcmc_step, seed + r)
        for th in draws:
            D.append((th, [float(v) for v in simulator(th, normal)]))
        train_flow(flow, D, epochs, lr, seed + r, batch=None)

        def logpost(th, _f=flow):
            lp = log_prior(th)
            if lp == float("-inf"):
                return lp
            return lp + flow_logprob(_f, x_o, th)

        history.append({"round": r + 1, "n_total": len(D),
                        "acceptance": rate,
                        "loglik_at_xo": sum(flow_logprob(flow, x_o, t)
                                            for t, _x in D[-10:]) / 10.0})
    post, rate = mcmc_sample(logpost, theta0, int(n_posterior), mcmc_burn,
                             mcmc_step, seed + 999)
    d = len(theta0)
    m = [sum(p[i] for p in post) / len(post) for i in range(d)]
    v = [sum((p[i] - m[i]) ** 2 for p in post) / max(len(post) - 1.0, 1.0)
         for i in range(d)]
    return RichResult(payload={
        "estimate": m,
        "posterior_mean": m,
        "posterior_sd": [math.sqrt(x) for x in v],
        "posterior_samples": post,
        "flow": flow,
        "D": D,
        "n_simulations": len(D),
        "history": history,
        "acceptance": rate,
        "n_rounds": int(n_rounds),
        "method": ("Sequential Neural Likelihood (Papamakarios, Sterratt "
                   "& Murray 2019) with a Masked Autoregressive Flow"),
        "note": ("Algorithm 1 retrains on the whole of D each round, not "
                 "the newest round, and proposes from the current "
                 "posterior estimate; round 1 proposes from the prior "
                 "since p_hat_0 = p(theta). Flow gradients are central "
                 "differences, which is slow and avoids a hand-rolled "
                 "backward pass"),
    })


sequential_neural_likelihood = abcnnt


def cheatsheet():
    return ("abcnnt: Sequential Neural Likelihood (Papamakarios, "
            "Sterratt & Murray 2019). Instead of rejecting simulations "
            "like ABC, learn the likelihood: each round draws theta from "
            "the current posterior estimate by MCMC, simulates x, adds "
            "the pair to D, retrains a Masked Autoregressive Flow "
            "q(x|theta) on ALL of D, and sets the posterior estimate to "
            "q(x_o|theta)p(theta). The flow is a stack of affine "
            "autoregressive transforms, so log q = log N(u;0,I) - sum "
            "alpha_i by the change of variables.")

# public names resolved by fn/_lazy_map.json
abc_neural = abcnnt
abcneural = abcnnt
