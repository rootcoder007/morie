# morie.fn -- function file (rootcoder007/morie)
"""MuZero world model: representation plus dynamics rollout."""

from ._richresult import RichResult

__all__ = ["mzworld", "muzero_world_model"]


def mzworld(observation, actions, representation, dynamics):
    """Roll the learned model forward from an observation.

    The representation function maps past observations to the root hidden
    state, and the dynamics function then advances it one hypothetical
    step per action while emitting the predicted immediate reward:

        s^0      = h(o_1, ..., o_t)
        r^k, s^k = g(s^{k-1}, a^k),   k = 1..K

    The hidden states carry no imposed semantics; nothing constrains them
    to match the environment's own state, which is the point of the
    construction.

    Parameters
    ----------
    observation : object
        Observation history passed straight to ``representation``.
    actions : sequence
        Actions a^1..a^K.
    representation : callable
        h, mapping the observation history to the root hidden state.
    dynamics : callable
        g, mapping (state, action) to the pair (reward, next state).

    Returns
    -------
    RichResult
        ``states`` (s^0..s^K), ``rewards`` (r^1..r^K), ``root``, ``K``.

    References
    ----------
    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
    Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D.,
    Graepel, T., Lillicrap, T. and Silver, D. (2020), "Mastering Atari,
    Go, chess and shogi by planning with a learned model", Nature 588,
    604-609; arXiv:1911.08265.  Read from the ar5iv rendering of the
    arXiv source.  Equation (1) unrolls s^0 = h_theta(o_1..o_t) and r^k, s^k =
    g_theta(s^{k-1}, a^k) for K hypothetical steps; the Methods note that
    the hidden state is not required to represent environment state.
    """
    s0 = representation(observation)
    s = s0
    states = [s0]
    rewards = []
    for a in actions:
        r, s = dynamics(s, a)
        rewards.append(float(r))
        states.append(s)
    return RichResult(payload={
        "states": states, "rewards": rewards, "root": s0,
        "K": len(rewards),
        "method": "MuZero world-model rollout (Schrittwieser et al. 2020 eq. 1)"})


muzero_world_model = mzworld


def cheatsheet():
    return "agmuzu: MuZero world model: representation plus dynamics rollout."
