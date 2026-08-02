# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dueling DQN value/advantage decomposition."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dueling_dqn"]

_METHOD = "Dueling DQN Q = V + (A - mean A)"


def geron_dueling_dqn(V, A):
    r"""Recombine a state value and per-action advantages.

    .. math::
        Q(s, a) = V(s) + \Bigl(A(s, a)
        - \frac{1}{|\mathcal A|}\sum_{a'} A(s, a')\Bigr)

    Subtracting the mean advantage is not cosmetic: without it the
    decomposition is unidentifiable -- add a constant to ``V`` and take
    it off ``A`` and ``Q`` is unchanged, so nothing pins the two heads
    down.  Forcing the advantages to average to zero fixes the gauge and
    makes ``V`` the actual state value.

    What the split buys is sample efficiency: ``V`` is learned from
    every transition in a state regardless of which action was taken,
    which matters in the many states where the action barely matters.

    Parameters
    ----------
    V : array-like, shape (n_states,) or scalar
        State values.
    A : array-like, shape (n_states, n_actions) or (n_actions,)
        Raw advantage-stream outputs.

    Returns
    -------
    RichResult
        Payload keys ``Q``, ``centered_advantage``, ``mean_advantage``,
        ``best_action``, ``advantage_sums_to_zero``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Dueling DQN section (Wang et al. 2016).

    Examples
    --------
    Advantages ``[1, 3]`` have mean 2, so they centre to ``[-1, 1]`` and
    a state value of 1 gives ``Q = [0, 2]``:

    >>> r = geron_dueling_dqn([1.0], [[1.0, 3.0]])
    >>> r["Q"]
    [[0.0, 2.0]]
    >>> r["centered_advantage"]
    [[-1.0, 1.0]]
    >>> r["advantage_sums_to_zero"]
    True

    Shifting every raw advantage by the same amount changes nothing --
    which is exactly the ambiguity the centering removes:

    >>> geron_dueling_dqn([1.0], [[101.0, 103.0]])["Q"]
    [[0.0, 2.0]]

    The mean of ``Q`` over actions is ``V``, by construction:

    >>> r2 = geron_dueling_dqn([5.0], [[0.0, 2.0, 7.0]])
    >>> round(sum(r2["Q"][0]) / 3, 10)
    5.0
    """
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    if Am.ndim != 2 or Am.size == 0:
        raise ValueError(f"A must be a non-empty (n_states, n_actions) array, got {Am.shape}.")
    if not np.all(np.isfinite(Am)):
        raise ValueError("A must be finite.")
    Vv = np.atleast_1d(np.asarray(V, dtype=float)).ravel()
    if Vv.size == 1 and Am.shape[0] != 1:
        Vv = np.full(Am.shape[0], float(Vv[0]))
    if Vv.size != Am.shape[0]:
        raise ValueError(
            f"V has {Vv.size} entries but A covers {Am.shape[0]} states."
        )
    if not np.all(np.isfinite(Vv)):
        raise ValueError("V must be finite.")
    if Am.shape[1] < 2:
        raise ValueError(
            f"the decomposition needs at least 2 actions to have a meaningful "
            f"advantage spread, got {Am.shape[1]}."
        )

    mean_a = Am.mean(axis=1, keepdims=True)
    cent = Am - mean_a
    Q = Vv[:, None] + cent

    return RichResult(
        title="Dueling DQN",
        summary_lines=[("States", int(Am.shape[0])), ("Actions", int(Am.shape[1]))],
        payload={
            "Q": Q.tolist(),
            "centered_advantage": cent.tolist(),
            "mean_advantage": mean_a.ravel().tolist(),
            "best_action": Q.argmax(axis=1).tolist(),
            "advantage_sums_to_zero": bool(np.allclose(cent.sum(axis=1), 0.0, atol=1e-12)),
            "estimate": Q.tolist(),
            "n": int(Am.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grduel: Q = V + (A - mean A); the mean subtraction fixes the V/A gauge"
