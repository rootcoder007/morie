# morie.fn -- function file (rootcoder007/morie)
"""MuZero recurrent inference step."""

from ._richresult import RichResult

__all__ = ["mzrecur", "muzero_recurrent_inf"]


def mzrecur(state, action, dynamics, prediction=None):
    """One recurrent-inference step of the learned model.

    MuZero's model has three learned components: a representation h, a
    dynamics g and a prediction f.  Recurrent inference applies the last
    two, taking the hidden state produced at the previous hypothetical
    step and the action chosen there,

        r^k, s^k = g(s^{k-1}, a^k)
        p^k, v^k = f(s^k)

    Neither g nor f is fixed by the algorithm, so both are arguments; the
    routine owns the wiring, which is the part that is the method.

    Parameters
    ----------
    state : object
        Hidden state s^{k-1}.
    action : object
        Action a^k.
    dynamics : callable
        g, mapping (state, action) to the pair (reward, next state).
    prediction : callable or None
        f, mapping a state to the pair (policy, value).  ``None`` skips
        the prediction head.

    Returns
    -------
    RichResult
        ``state``, ``reward``, ``policy``, ``value``.

    References
    ----------
    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
    Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D.,
    Graepel, T., Lillicrap, T. and Silver, D. (2020), "Mastering Atari,
    Go, chess and shogi by planning with a learned model", Nature 588,
    604-609; arXiv:1911.08265.  Read from the ar5iv rendering of the
    arXiv source.  The Search paragraph of the Methods gives the expansion step
    r^l, s^l = g_theta(s^{l-1}, a^l) followed by p^l, v^l =
    f_theta(s^l); Equation (1) is the same model unrolled for K steps.
    """
    r, s = dynamics(state, action)
    p, v = (None, None) if prediction is None else prediction(s)
    return RichResult(payload={
        "state": s, "reward": float(r), "policy": p,
        "value": None if v is None else float(v),
        "method": "MuZero recurrent inference (Schrittwieser et al. 2020)"})


muzero_recurrent_inf = mzrecur


def cheatsheet():
    return "agmurc: MuZero recurrent inference step."
