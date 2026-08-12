r"""Direct Preference Optimization: loss, implicit rewards, gradient weights.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., &
Finn, C. (2023) "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model", arXiv:2305.18290.

The KL-constrained RLHF objective has the closed-form optimum (eq. 4)

.. math:: \pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x)
          \exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big),

which rearranges (eq. 5) to express the reward in terms of its own
optimal policy,

.. math:: r(x, y) = \beta \log
          \frac{\pi_r(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
          + \beta \log Z(x).

Because Bradley-Terry depends only on reward *differences*, the
intractable :math:`\beta \log Z(x)` cancels and the preference
likelihood becomes a function of the policy alone. Maximum likelihood
on that gives the DPO loss (eq. 7):

.. math:: \mathcal{L}_{\mathrm{DPO}} = -\,\mathbb{E}\Big[
          \log \sigma\Big(\beta \log
          \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
          - \beta \log
          \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
          \Big)\Big].

The *implicit reward* is :math:`\hat r_\theta(x,y) = \beta \log
\pi_\theta(y \mid x)/\pi_{\mathrm{ref}}(y \mid x)`, and the gradient is

.. math:: \nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta\,
          \mathbb{E}\big[\sigma(\hat r_\theta(x,y_l)
          - \hat r_\theta(x,y_w))\,
          (\nabla_\theta \log \pi(y_w \mid x)
          - \nabla_\theta \log \pi(y_l \mid x))\big].

That leading :math:`\sigma(\cdot)` is the per-pair weight the paper
calls out: it is large exactly when the implicit reward model has the
pair the wrong way round. It is returned here as ``grad_weight``
because it is the diagnostic worth looking at, and because the paper
reports (Appendix Table 3) that dropping it degenerates the model.

Both preference models in the paper are implemented:

``model="bradley-terry"``
    Eq. 7, pairwise. ``logp_*`` are the sequence log-probabilities
    :math:`\log \pi(y \mid x)` of the chosen (``_w``) and rejected
    (``_l``) completions under the policy and the reference.

``model="plackett-luce"``
    Eq. 20, the ranking generalisation. Each row is a full ranking of
    :math:`K` completions, best first; the loss is

    .. math:: -\log \prod_{k=1}^{K}
              \frac{\exp \hat r_{\tau(k)}}
                   {\sum_{j=k}^{K} \exp \hat r_{\tau(j)}}.

    At :math:`K = 2` eq. 20 reduces to eq. 7 exactly (the paper says
    so at eq. 18; the anchors check it numerically).

This module scores and differentiates the DPO objective; it does not
fine-tune a language model. The inputs are log-probabilities, which is
the interface at which the objective is actually defined -- everything
above the log-probabilities is model plumbing, everything below is this
file.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["dpoF", "dpo_loss", "dpoloss"]

_MODELS = ("bradley-terry", "plackett-luce")


def _logsigmoid(z):
    """log sigma(z), stable in both tails."""
    if z >= 0.0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logsumexp(vals):
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _vec(x, name):
    v = [float(t) for t in np.atleast_1d(np.asarray(x, dtype=float))]
    if not v:
        raise ValueError("dpoF: %s must be non-empty" % name)
    return v


def dpoF(logp_w=None, logp_l=None, logp_ref_w=None, logp_ref_l=None,
         beta=0.1, model="bradley-terry", logp=None, logp_ref=None,
         label_smoothing=0.0):
    r"""Evaluate the DPO objective and its per-pair gradient weights.

    Parameters
    ----------
    logp_w, logp_l : array-like
        :math:`\log \pi_\theta(y_w \mid x)` and
        :math:`\log \pi_\theta(y_l \mid x)`, one entry per preference
        pair. Bradley-Terry only.
    logp_ref_w, logp_ref_l : array-like
        The same quantities under :math:`\pi_{\mathrm{ref}}`.
    beta : float
        :math:`\beta`, the strength of the KL constraint in eq. 3.
        Must be > 0.
    model : {"bradley-terry", "plackett-luce"}
        Eq. 7 or eq. 20.
    logp, logp_ref : array-like
        Plackett-Luce only: ``(n_rankings, K)`` log-probabilities, with
        each row already ordered best-first (i.e. ordered by
        :math:`\tau`).
    label_smoothing : float
        Optional :math:`\varepsilon \in [0, 0.5)` mixing weight on the
        flipped pair, i.e. the conservative variant
        :math:`-(1-\varepsilon)\log \sigma(z) - \varepsilon
        \log \sigma(-z)`. Zero (the paper's own objective) by default.
        Bradley-Terry only.

    Returns
    -------
    RichResult
        ``estimate`` / ``loss`` is the mean loss. Also
        ``losses`` (per pair or per ranking), ``reward_w`` /
        ``reward_l`` / ``margin`` (the implicit rewards
        :math:`\hat r_\theta` and their difference), ``grad_weight``
        (:math:`\sigma(\hat r_l - \hat r_w)`), and ``accuracy``, the
        fraction of pairs the implicit reward model already orders
        correctly.

    References
    ----------
    Rafailov et al. (2023) arXiv:2305.18290, eqs. 4, 5, 7, 18, 19, 20
    and the gradient expression of section 4.
    """
    if model not in _MODELS:
        raise ValueError("dpoF: model must be one of %r, got %r"
                         % (_MODELS, model))
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("dpoF: beta must be > 0, got %r" % (beta,))

    if model == "plackett-luce":
        return _plackett_luce(logp, logp_ref, beta)

    eps = float(label_smoothing)
    if not 0.0 <= eps < 0.5:
        raise ValueError("dpoF: label_smoothing must lie in [0, 0.5), "
                         "got %r" % (eps,))
    pw = _vec(logp_w, "logp_w")
    pl = _vec(logp_l, "logp_l")
    rw = _vec(logp_ref_w, "logp_ref_w")
    rl = _vec(logp_ref_l, "logp_ref_l")
    n = len(pw)
    if not (len(pl) == len(rw) == len(rl) == n):
        raise ValueError("dpoF: logp_w, logp_l, logp_ref_w and logp_ref_l "
                         "must have the same length")

    reward_w = [beta * (pw[i] - rw[i]) for i in range(n)]
    reward_l = [beta * (pl[i] - rl[i]) for i in range(n)]
    margin = [reward_w[i] - reward_l[i] for i in range(n)]
    if eps == 0.0:
        losses = [-_logsigmoid(m) for m in margin]
    else:
        losses = [-(1.0 - eps) * _logsigmoid(m) - eps * _logsigmoid(-m)
                  for m in margin]
    grad_w = [_sigmoid(-m) for m in margin]
    acc = sum(1.0 for m in margin if m > 0.0) / n

    loss = sum(losses) / n
    return RichResult(payload={
        "estimate": float(loss),
        "loss": float(loss),
        "losses": losses,
        "reward_w": reward_w,
        "reward_l": reward_l,
        "margin": margin,
        "grad_weight": grad_w,
        "accuracy": float(acc),
        "beta": beta,
        "n": n,
        "model": "bradley-terry",
        "method": "DPO (Rafailov et al. 2023 eq. 7)",
    })


def _plackett_luce(logp, logp_ref, beta):
    r"""Eq. 20. Rows are rankings, already ordered best-first."""
    if logp is None or logp_ref is None:
        raise ValueError("dpoF: model='plackett-luce' needs logp and "
                         "logp_ref, shape (n_rankings, K)")
    P = [[float(v) for v in row]
         for row in np.atleast_2d(np.asarray(logp, dtype=float))]
    R = [[float(v) for v in row]
         for row in np.atleast_2d(np.asarray(logp_ref, dtype=float))]
    if len(P) != len(R):
        raise ValueError("dpoF: logp and logp_ref must have the same "
                         "number of rankings")
    losses = []
    rewards = []
    for i, row in enumerate(P):
        K = len(row)
        if K < 2:
            raise ValueError("dpoF: each ranking needs K >= 2 completions")
        if len(R[i]) != K:
            raise ValueError("dpoF: ranking %d has %d policy entries but %d "
                             "reference entries" % (i, K, len(R[i])))
        rhat = [beta * (row[k] - R[i][k]) for k in range(K)]
        rewards.append(rhat)
        # log prod_k exp(r_k) / sum_{j>=k} exp(r_j)
        ll = 0.0
        for k in range(K):
            ll += rhat[k] - _logsumexp(rhat[k:])
        losses.append(-ll)
    loss = sum(losses) / len(losses)
    return RichResult(payload={
        "estimate": float(loss),
        "loss": float(loss),
        "losses": losses,
        "rewards": rewards,
        "beta": beta,
        "n": len(losses),
        "model": "plackett-luce",
        "method": "DPO (Rafailov et al. 2023 eq. 20)",
    })


def optimal_policy(logp_ref, reward, beta):
    r"""Eq. 4: the KL-constrained optimum
    :math:`\pi_r \propto \pi_{\mathrm{ref}} \exp(r/\beta)`.

    Returns normalised log-probabilities over the given support, and the
    log partition function :math:`\log Z(x)`. Eq. 5 is the inverse of
    this map, so ``beta * (optimal_policy(...) - logp_ref)`` recovers
    ``reward`` up to the additive constant :math:`\beta \log Z`; the
    anchors check exactly that round trip.
    """
    lr = _vec(logp_ref, "logp_ref")
    rr = _vec(reward, "reward")
    if len(lr) != len(rr):
        raise ValueError("optimal_policy: logp_ref and reward must have "
                         "the same length")
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("optimal_policy: beta must be > 0")
    unnorm = [lr[i] + rr[i] / beta for i in range(len(lr))]
    logZ = _logsumexp(unnorm)
    return [u - logZ for u in unnorm], float(logZ)


def cheatsheet():
    return ("dpoF: DPO loss -log sigma(beta log pi_w/ref_w - beta log "
            "pi_l/ref_l) (Rafailov 2023 eq. 7); implicit reward "
            "rhat = beta log pi/pi_ref; grad weight sigma(rhat_l - "
            "rhat_w); model='plackett-luce' is eq. 20 and reduces to "
            "eq. 7 at K=2. optimal_policy() is eq. 4.")


# compact aliases per ledger/NAMING.md
dpo_loss = dpoF
dpoloss = dpoF
