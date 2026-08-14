"""Monte Carlo tree search with UCT (Coulom 2006; Browne et al. 2012)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mctsr", "monte_carlo_tree_search"]


class _Node:
    __slots__ = ("state", "parent", "action", "children", "untried",
                 "N", "Q")

    def __init__(self, state, parent, action, untried):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried = list(untried)
        self.N = 0
        self.Q = 0.0


def _best_child(node, c, rng):
    # UCT (Browne et al. Eq. for UCT):
    #   UCT_j = Xbar_j + 2 Cp sqrt(2 ln n / n_j),
    # n = parent visits, n_j = child visits.  Unvisited children
    # score +inf.  "If more than one child node has the same maximal
    # value, the tie is usually broken randomly" -- done here with
    # the native RNG so both language arms agree.
    best, ties = -math.inf, []
    for ch in node.children:
        if ch.N == 0:
            v = math.inf
        else:
            v = ch.Q / ch.N + 2.0 * c * math.sqrt(
                2.0 * math.log(node.N) / ch.N)
        if v > best:
            best, ties = v, [ch]
        elif v == best:
            ties.append(ch)
    if len(ties) == 1:
        return ties[0]
    return ties[int(float(rng.uniform()) * len(ties))]


def mctsr(root_state, actions, step, reward, is_terminal,
          n_iter=200, c=0.7071067811865476, seed=0,
          backup="sum", final="robust"):
    """
    Vanilla Monte Carlo tree search with UCT and random rollouts.

    Coulom (2006) introduced MCTS; Browne et al. (2012) give the
    canonical statement used here.  Each iteration runs their
    Algorithm 2: TREEPOLICY descends by BESTCHILD while the node is
    fully expanded, EXPAND adds one untried action, DEFAULTPOLICY
    plays uniformly at random to a terminal state, and BACKUP
    propagates the reward to the root.  Selection uses

        UCT_j = Xbar_j + 2 Cp sqrt( 2 ln n / n_j ),

    with n the parent's visit count, n_j the child's, and unvisited
    children valued at infinity.  The default Cp = 1/sqrt(2) is the
    value Kocsis & Szepesvari proved satisfies the Hoeffding
    inequality for rewards in [0, 1] (survey Sec. 3.3); the survey
    notes a different Cp may be needed for rewards outside [0, 1].

    Both backup rules in the survey are provided.  backup="sum"
    (default) is Algorithm 2: N(v) += 1, Q(v) += Delta, unchanged up
    the path -- correct for single-agent and general reward vectors.
    backup="negamax" is their Algorithm 3, the two-player zero-sum
    variant that negates Delta at each level.

    Both final-move criteria the survey names are provided.
    final="robust" (default) returns the MOST-VISITED root child,
    which is BESTCHILD(v0, 0) in Algorithm 2 once visit counts and
    values agree, and is the standard robust choice; final="max"
    returns the highest mean-reward child.

    Sources
    -------
    Browne, C. B. et al. (2012).  A survey of Monte Carlo tree
    search methods.  *IEEE Transactions on Computational
    Intelligence and AI in Games* 4(1), 1-43, Sec. 3.3 (UCT),
    Algorithm 2 (UCT search) and Algorithm 3 (negamax backup)
    (local copy fetched-wave3/MCTS_survey_Browne_2012.pdf).
    Coulom, R. (2006).  Efficient selectivity and backup operators
    in Monte-Carlo tree search.  *Computers and Games 2006*, 72-83.

    Parameters
    ----------
    root_state : object
        Starting state (must be hashable/copyable by the callbacks).
    actions : callable
        actions(state) -> list of legal actions.
    step : callable
        step(state, action) -> next state.
    reward : callable
        reward(terminal_state) -> float, ideally in [0, 1].
    is_terminal : callable
        is_terminal(state) -> bool.
    n_iter : int
        Number of MCTS iterations (the "computational budget").
    c : float
        Exploration constant Cp (default 1/sqrt(2)).
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).
    backup : {"sum", "negamax"}
        Algorithm 2 or Algorithm 3 backup.
    final : {"robust", "max"}
        Most-visited or highest-mean root child.

    Returns
    -------
    RichResult
        Keys: action, root_visits, child_visits, child_values,
        n_iter, c, backup, final.
    """
    if backup not in ("sum", "negamax"):
        raise ValueError("backup must be 'sum' or 'negamax'")
    if final not in ("robust", "max"):
        raise ValueError("final must be 'robust' or 'max'")
    rng = np.random.default_rng(seed)
    root = _Node(root_state, None, None, actions(root_state))
    if not root.untried and not is_terminal(root_state):
        raise ValueError("root has no legal actions")

    for _ in range(int(n_iter)):
        # --- TREEPOLICY ---
        v = root
        while not is_terminal(v.state):
            if v.untried:
                a = v.untried.pop(0)
                s2 = step(v.state, a)
                ch = _Node(s2, v, a, actions(s2))
                v.children.append(ch)
                v = ch
                break
            if not v.children:
                break
            v = _best_child(v, c, rng)
        # --- DEFAULTPOLICY: uniformly random to a terminal state ---
        s = v.state
        while not is_terminal(s):
            acts = actions(s)
            if not acts:
                break
            s = step(s, acts[int(float(rng.uniform()) * len(acts))])
        delta = float(reward(s))
        # --- BACKUP ---
        node = v
        while node is not None:
            node.N += 1
            node.Q += delta
            if backup == "negamax":
                delta = -delta
            node = node.parent

    if not root.children:
        raise ValueError("no children expanded; increase n_iter")
    if final == "robust":
        best = max(root.children, key=lambda ch: (ch.N, -id(ch)))
    else:
        best = max(root.children,
                   key=lambda ch: (ch.Q / ch.N if ch.N else -math.inf,
                                   -id(ch)))
    return RichResult(payload={
        "action": best.action,
        "root_visits": root.N,
        "child_visits": {str(ch.action): ch.N for ch in root.children},
        "child_values": {str(ch.action): (ch.Q / ch.N if ch.N else 0.0)
                         for ch in root.children},
        "n_iter": int(n_iter),
        "c": float(c),
        "backup": backup,
        "final": final,
        "seed": int(seed),
        "method": "UCT MCTS (Browne et al. 2012, Algorithm %s)"
                  % ("2" if backup == "sum" else "3"),
    })


# long descriptive alias (stub-era name)
monte_carlo_tree_search = mctsr


def cheatsheet():
    return ("mctsr: UCT = Xbar_j + 2*Cp*sqrt(2*ln n / n_j); "
            "Cp=1/sqrt(2); robust=most-visited root child")

# public names resolved by fn/_lazy_map.json
mcts_rollout = mctsr
mctsrollout = mctsr
