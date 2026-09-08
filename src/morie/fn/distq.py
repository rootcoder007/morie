# morie.fn -- function file (rootcoder007/morie)
r"""C51: the categorical distributional Bellman update.

Ordinary value-based reinforcement learning tracks :math:`E[Z]`, the
expected return. C51 tracks the *distribution* of the return on a fixed
grid of :math:`N` atoms,

.. math:: z_i = V_{\min} + i\,\Delta z, \qquad
          \Delta z = \frac{V_{\max} - V_{\min}}{N - 1},

with the network emitting a softmax :math:`p_i(x, a)` over them.

**The supports do not line up, and that is the whole problem.** Applying
the Bellman operator maps each atom to :math:`\hat{\mathcal{T}}z_j =
r + \gamma z_j`, which lands between atoms. The fix is to project the
updated distribution back onto the grid, splitting each atom's
probability between its two neighbours in proportion to closeness:

.. math:: (\Phi\hat{\mathcal{T}}Z_\theta(x,a))_i
          = \sum_{j=0}^{N-1}
            \Big[1 - \frac{|[\hat{\mathcal{T}}z_j]_{V_{\min}}^{V_{\max}}
                            - z_i|}{\Delta z}\Big]_0^1
            p_j(x', \pi(x')),

which turns the Bellman update into multiclass classification: the loss
is the cross-entropy of the projected target against the current
distribution.

**The exact-hit case is where implementations lose mass.** Algorithm 1
computes :math:`b_j = (\hat{\mathcal{T}}z_j - V_{\min})/\Delta z` and
splits :math:`p_j` as :math:`(u - b_j)` to the lower atom and
:math:`(b_j - l)` to the upper. When :math:`b_j` lands exactly on an
atom, :math:`l = u = b_j` and *both* those factors are zero -- written
literally, the atom's entire probability silently disappears. It is easy
to miss because it only fires on exact hits, which is precisely what
happens when :math:`\gamma = 0`, or when rewards and atoms share a
lattice. The projection below adds the full mass when ``l == u``, and
the anchor checks the total is 1 in exactly those cases.

**Clipping is not optional either.** :math:`\hat{\mathcal{T}}z_j` is
bounded to :math:`[V_{\min}, V_{\max}]` before projecting, so returns
outside the representable range pile onto the boundary atom rather than
falling off the grid.

References
----------
Bellemare, M. G., Dabney, W. & Munos, R. (2017) "A Distributional
Perspective on Reinforcement Learning", *Proceedings of the 34th
International Conference on Machine Learning*, PMLR 70, 449-458,
arXiv:1707.06887. Sec. 4.1-4.2, eq. (7), and Algorithm 1.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J.,
Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K.,
Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I.,
King, H., Kumaran, D., Wierstra, D., Legg, S. & Hassabis, D. (2015)
"Human-level control through deep reinforcement learning", *Nature*
518(7540), 529-533, doi:10.1038/nature14236. The DQN architecture C51
modifies, and the target-network convention.

Rowland, M., Bellemare, M. G., Dabney, W., Munos, R. & Teh, Y. W. (2018)
"An Analysis of Categorical Distributional Reinforcement Learning",
*Proceedings of the 21st International Conference on Artificial
Intelligence and Statistics*, PMLR 84, 29-37, arXiv:1802.08163. Shows
the projected operator is a contraction in Cramer distance, which is why
the iteration in :func:`value_distribution_iteration` converges.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["atoms", "categorical_projection", "categorical_loss",
           "greedy_action", "c51_update", "bernoulli_algorithm",
           "value_distribution_iteration", "distribution_mean"]

_EPS = 1e-12


def atoms(v_min, v_max, n_atoms):
    """The fixed support {z_i} and its spacing."""
    n = int(n_atoms)
    if n < 2:
        raise ValueError("distq: need at least 2 atoms, got %d" % n)
    lo, hi = float(v_min), float(v_max)
    if not hi > lo:
        raise ValueError("distq: need v_max > v_min, got %r and %r"
                         % (v_min, v_max))
    dz = (hi - lo) / (n - 1)
    return [lo + i * dz for i in range(n)], dz


def distribution_mean(probs, z):
    """E[Z] = sum_i z_i p_i."""
    if len(probs) != len(z):
        raise ValueError("distq: %d probabilities for %d atoms"
                         % (len(probs), len(z)))
    return sum(z[i] * probs[i] for i in range(len(z)))


def categorical_projection(reward, gamma, next_probs, v_min, v_max,
                           n_atoms=None, done=False):
    r"""Eq. (7) / Algorithm 1: project r + gamma*z onto {z_i}.

    Parameters
    ----------
    reward : float
        The sampled reward r_t.
    gamma : float
        The discount for this transition, in [0, 1]. Pass 0 (or
        ``done=True``) for a terminal transition, which collapses the
        target onto the atoms nearest r.
    next_probs : array-like
        p_j(x', pi(x')), the next state's distribution under the greedy
        action. Must be a probability vector.
    done : bool
        Terminal flag; equivalent to gamma = 0.

    Returns
    -------
    list
        The projected probabilities m_i, which sum to 1.
    """
    p = [float(v) for v in next_probs]
    n = len(p) if n_atoms is None else int(n_atoms)
    if len(p) != n:
        raise ValueError("distq: %d next probabilities for %d atoms"
                         % (len(p), n))
    if any(v < -1e-9 for v in p):
        raise ValueError("distq: next_probs has a negative entry")
    tot = sum(p)
    if abs(tot - 1.0) > 1e-6:
        raise ValueError("distq: next_probs sums to %.9f, not 1" % tot)
    g = 0.0 if done else float(gamma)
    if not 0.0 <= g <= 1.0:
        raise ValueError("distq: gamma must be in [0, 1], got %r"
                         % (gamma,))
    z, dz = atoms(v_min, v_max, n)
    m = [0.0] * n
    for j in range(n):
        # T_hat z_j, clipped to the representable range
        tz = min(max(float(reward) + g * z[j], z[0]), z[-1])
        b = (tz - z[0]) / dz
        lo = int(math.floor(b))
        hi = int(math.ceil(b))
        lo = min(max(lo, 0), n - 1)
        hi = min(max(hi, 0), n - 1)
        if lo == hi:
            # b landed exactly on an atom. Written literally, Algorithm
            # 1 would add p*(u - b) + p*(b - l) = 0 here and lose the
            # mass entirely.
            m[lo] += p[j]
        else:
            m[lo] += p[j] * (hi - b)
            m[hi] += p[j] * (b - lo)
    return m


def categorical_loss(m, probs, eps=1e-12):
    """The cross-entropy term of D_KL(Phi T_hat Z || Z): -sum m log p."""
    if len(m) != len(probs):
        raise ValueError("distq: %d targets for %d probabilities"
                         % (len(m), len(probs)))
    tot = 0.0
    for i in range(len(m)):
        tot -= m[i] * math.log(max(float(probs[i]), eps))
    return tot


def greedy_action(next_probs_by_action, z):
    """a* = argmax_a sum_i z_i p_i(x', a), the first line of Algorithm 1."""
    if not next_probs_by_action:
        raise ValueError("distq: no actions given")
    qs = [distribution_mean([float(v) for v in row], z)
          for row in next_probs_by_action]
    best = max(range(len(qs)), key=lambda a: qs[a])
    return best, qs


def c51_update(reward, gamma, next_probs_by_action, current_probs,
               v_min, v_max, done=False):
    r"""Algorithm 1 end to end: greedy action, projection, loss.

    Returns the projected target ``m``, the cross-entropy loss, the
    chosen action and the Q values it was chosen by.
    """
    cur = [float(v) for v in current_probs]
    n = len(cur)
    z, _ = atoms(v_min, v_max, n)
    a_star, qs = greedy_action(next_probs_by_action, z)
    m = categorical_projection(reward, gamma,
                               next_probs_by_action[a_star], v_min,
                               v_max, n_atoms=n, done=done)
    loss = categorical_loss(m, cur)
    return RichResult(payload={
        "estimate": loss, "loss": loss, "target": m,
        "action": a_star, "q_values": qs,
        "q_target": distribution_mean(m, z),
        "q_current": distribution_mean(cur, z),
        "atoms": z, "n_atoms": n,
        "method": "categorical algorithm (C51), Bellemare, Dabney & "
                  "Munos (2017) Algorithm 1",
    })


def bernoulli_algorithm(reward, gamma, next_probs, v_min, v_max,
                        done=False):
    r"""The one-parameter N = 2 alternative the paper names.

    :math:`\Phi\hat{\mathcal{T}}Z := [(E[\hat{\mathcal{T}}Z] -
    V_{\min})/\Delta z]_0^1`, i.e. the target is a single Bernoulli
    parameter rather than a full histogram.
    """
    p = [float(v) for v in next_probs]
    z, dz = atoms(v_min, v_max, len(p))
    g = 0.0 if done else float(gamma)
    if not 0.0 <= g <= 1.0:
        raise ValueError("distq: gamma must be in [0, 1], got %r"
                         % (gamma,))
    ex = float(reward) + g * distribution_mean(p, z)
    return min(max((ex - v_min) / dz, 0.0), 1.0)


def value_distribution_iteration(reward_atoms, reward_probs, gamma,
                                 v_min, v_max, n_atoms, iters=400,
                                 tol=1e-13):
    r"""Iterate the projected operator on a single self-looping state.

    The return is :math:`Z = R + \gamma Z'` with R drawn afresh each
    step, so this converges to the stationary value distribution. The
    projected operator is a contraction in Cramer distance (Rowland et
    al. 2018), which is why iterating it converges at all.
    """
    ra = [float(v) for v in reward_atoms]
    rp = [float(v) for v in reward_probs]
    if len(ra) != len(rp):
        raise ValueError("distq: %d reward atoms but %d probabilities"
                         % (len(ra), len(rp)))
    if abs(sum(rp) - 1.0) > 1e-9:
        raise ValueError("distq: reward_probs sums to %.9f, not 1"
                         % sum(rp))
    z, _ = atoms(v_min, v_max, n_atoms)
    n = len(z)
    cur = [1.0 / n] * n
    for step in range(int(iters)):
        nxt = [0.0] * n
        for t in range(len(ra)):
            proj = categorical_projection(ra[t], gamma, cur, v_min,
                                          v_max, n_atoms=n)
            for i in range(n):
                nxt[i] += rp[t] * proj[i]
        shift = max(abs(nxt[i] - cur[i]) for i in range(n))
        cur = nxt
        if shift < tol:
            return cur, {"iterations": step + 1, "converged": True,
                         "shift": shift}
    return cur, {"iterations": int(iters), "converged": False,
                 "shift": shift}


def cheatsheet():
    return ("distq: C51. Atoms z_i on [v_min, v_max]; project "
            "T_hat z_j = clip(r + gamma z_j) onto the grid, splitting "
            "p_j between the two neighbours by (u-b) and (b-l); loss is "
            "the cross-entropy -sum m_i log p_i (Bellemare-Dabney-Munos "
            "2017 Alg. 1). When b lands EXACTLY on an atom, l == u and "
            "both split factors are zero -- add the full mass or it "
            "vanishes.")


# compact alias per ledger/NAMING.md
categoricalprojection = categorical_projection

# public names resolved by fn/_lazy_map.json
distributional_rl = categorical_projection
