"""AlphaDev: discovering a sorting routine as a sequence of machine
instructions, by search.

The trick of the paper is the reframing. Do not ask a model to write a
sorting algorithm. Ask it to play a one-player game -- the AssemblyGame
-- whose moves are single machine instructions and whose position is
the program written so far together with what that program does to a
set of test inputs when you actually run it. A move that makes the
outputs more sorted is a good move. The game ends when the program
sorts every test input, and the score then falls back on how few
instructions it took. What comes out is not a proof and not a
transcript of reasoning; it is a program, and it either sorts or it
does not.

Three pieces, all of them checkable:

  THE MACHINE. Registers and memory, and four instructions, which are
  the ones the paper works in: an unconditional move, a compare that
  sets a flag, and two moves conditional on that flag. Nothing here is
  a heuristic -- the machine is deterministic and its semantics are
  written out in ``step``. This is what makes the whole exercise
  falsifiable: a candidate program is not scored by resemblance to a
  sorting algorithm, it is EXECUTED.

  THE REWARD. The paper measures correctness as how much of the output
  is right, and latency by how long the program is. Both are here:
  ``correctness`` counts the memory slots holding the value they should
  hold, summed over the test inputs, and the score subtracts a weight
  times the instruction count. The weight is a parameter because the
  trade it governs -- one more instruction against one more correct
  element -- is the caller's to make, and because at weight zero the
  search is a pure correctness search, which is the honest baseline to
  compare against.

  THE SEARCH. Two routes, and the second exists to keep the first
  honest.

    ``mcts``  Monte Carlo tree search with the PUCT selection rule, the
              method the paper uses. AlphaDev guides it with a trained
              policy and value network; there are no weights here, so
              the prior is uniform over the legal moves and a position
              is valued by the correctness it has actually reached.
              That is a weaker search than the paper's, and it is
              labelled as such rather than dressed up.

    ``bfs``   Exhaustive breadth-first enumeration of every program up
              to the length limit. Exponential, so only usable on small
              action spaces -- but it returns the PROVABLE optimum, and
              a tree search that beats a provable optimum is a tree
              search with a bug. That comparison is the point of
              keeping it.

Both routes return the best program they saw, not the last one, because
a search that wandered away from a good program should still report it.

References
  Mankowitz, D.J., Michi, A., Zhernov, A., Gelmi, M., Selvi, M.,
    Paduraru, C., Leurent, E., Iqbal, S., Lespiau, J.-B., Ahern, A.,
    Koppe, T., Millikin, K., Gaffney, S., Elster, S., Broshear, J.,
    Gamble, C., Milan, K., Tung, R., Hwang, M., Cemgil, T., Barekatain,
    M., Li, Y., Mandhane, A., Hubert, T., Schrittwieser, J., Hassabis,
    D., Kohli, P., Riedmiller, M., Vinyals, O. and Silver, D. (2023)
    "Faster sorting algorithms discovered using deep reinforcement
    learning." Nature 618, 257-263. doi:10.1038/s41586-023-06004-9.
    The AssemblyGame, its correctness and latency rewards, and the
    instruction set.
  Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang,
    A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y.,
    Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G.,
    Graepel, T. and Hassabis, D. (2017) "Mastering the game of Go
    without human knowledge." Nature 550, 354-359. The PUCT selection
    rule used below.
  Rosin, C.D. (2011) "Multi-armed bandits with episode context."
    Annals of Mathematics and Artificial Intelligence 61(3), 203-230.
    Where PUCT comes from.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["alphadev_quicksort_disc", "assembly_run", "step",
           "correctness", "sort_action_space", "program_text",
           "cheatsheet"]

_OPS = ("mov", "cmp", "cmovl", "cmovg")


def _read(mem, reg, loc):
    bank, idx = loc[0], int(loc[1])
    if bank == "M":
        if idx < 0 or idx >= len(mem):
            raise ValueError("the instruction reads outside memory")
        return mem[idx]
    if bank == "R":
        if idx < 0 or idx >= len(reg):
            raise ValueError("the instruction reads a register that is "
                             "not there")
        return reg[idx]
    raise ValueError("a location is in memory or in a register, nothing "
                     "else")


def _write(mem, reg, loc, v):
    bank, idx = loc[0], int(loc[1])
    if bank == "M":
        if idx < 0 or idx >= len(mem):
            raise ValueError("the instruction writes outside memory")
        mem[idx] = v
    elif bank == "R":
        if idx < 0 or idx >= len(reg):
            raise ValueError("the instruction writes a register that is "
                             "not there")
        reg[idx] = v
    else:
        raise ValueError("a location is in memory or in a register, "
                         "nothing else")


def step(mem, reg, flag, instr):
    """Execute one instruction. Deterministic, and the whole semantics.

    ``mov src dst``    dst takes the value of src.
    ``cmp a b``        the flag becomes minus one, zero or one as a is
                       below, equal to or above b. Nothing else moves.
    ``cmovl src dst``  dst takes src only if the flag is negative.
    ``cmovg src dst``  dst takes src only if the flag is positive.

    The conditional moves are the reason this instruction set is worth
    searching at all: they let a program reorder two values without a
    branch, which is what the discovered routines exploit.
    """
    op = instr[0]
    if op not in _OPS:
        raise ValueError("unknown instruction: " + str(op))
    a, b = instr[1], instr[2]
    m = list(mem)
    r = list(reg)
    f = int(flag)
    if op == "mov":
        _write(m, r, b, _read(m, r, a))
    elif op == "cmp":
        x = _read(m, r, a)
        y = _read(m, r, b)
        f = -1 if x < y else (1 if x > y else 0)
    elif op == "cmovl":
        if f < 0:
            _write(m, r, b, _read(m, r, a))
    else:
        if f > 0:
            _write(m, r, b, _read(m, r, a))
    return m, r, f


def assembly_run(program, x, n_reg):
    """Run a program on one input vector; return the memory it leaves.

    Memory starts holding the input, the registers start at zero and
    the flag starts cleared, so a program's behaviour depends on
    nothing but the instructions it contains.
    """
    mem = [float(v) for v in x]
    reg = [0.0] * int(n_reg)
    flag = 0
    for instr in program:
        mem, reg, flag = step(mem, reg, flag, instr)
    return mem


def correctness(program, inputs, targets, n_reg):
    """How many memory slots end up holding the value they should.

    Summed over the test inputs, so a program that sorts two of three
    cases scores strictly between one that sorts none and one that
    sorts all. The paper's alternative measure -- the squared distance
    from the target -- rewards being close; this one rewards being
    right, and being right is what a sorting routine has to be.
    """
    got = 0
    for x, t in zip(inputs, targets):
        out = assembly_run(program, x, n_reg)
        for k in range(len(t)):
            if out[k] == t[k]:
                got += 1
    return got


def sort_action_space(n_mem, n_reg):
    """Every legal instruction over the given memory and registers.

    Ordered so the enumeration is reproducible: by operation in the
    order they are defined, then by source, then by destination. A
    compare of a location with itself is dropped -- its flag is always
    zero, so it can only waste an instruction.
    """
    locs = [("M", i) for i in range(int(n_mem))]
    locs += [("R", i) for i in range(int(n_reg))]
    out = []
    for op in _OPS:
        for a in locs:
            for b in locs:
                if a == b:
                    continue
                out.append((op, a, b))
    return out


def program_text(program):
    """A program as one readable line per instruction."""
    return "\n".join("%s %s%d %s%d" % (i[0], i[1][0], i[1][1],
                                       i[2][0], i[2][1])
                     for i in program)


def _score(program, inputs, targets, n_reg, latency_weight, reward_fn):
    c = (correctness(program, inputs, targets, n_reg) if reward_fn is None
         else float(reward_fn(program, inputs, targets, n_reg)))
    return float(c) - float(latency_weight) * len(program), float(c)


def _bfs(inputs, targets, action_space, n_reg, max_len, latency_weight,
         reward_fn):
    """Every program up to the length limit, in order. The optimum."""
    best = []
    best_s, best_c = _score([], inputs, targets, n_reg, latency_weight,
                            reward_fn)
    frontier = [[]]
    seen = 1
    for _ in range(int(max_len)):
        nxt = []
        for prog in frontier:
            for act in action_space:
                cand = prog + [act]
                seen += 1
                s, c = _score(cand, inputs, targets, n_reg,
                              latency_weight, reward_fn)
                if s > best_s:
                    best_s, best_c, best = s, c, cand
                nxt.append(cand)
        frontier = nxt
    return best, best_s, best_c, seen


def _mcts(inputs, targets, action_space, n_reg, max_len, latency_weight,
          reward_fn, n_sim, c_puct, seed):
    """PUCT tree search over programs.

    Nodes are keyed by the program that reaches them, so the tree is a
    tree and not a graph -- two different instruction orders reaching
    the same machine state are different nodes, which is what the game
    says. Priors are uniform because there is no trained policy here,
    and a leaf is valued by the correctness it has reached, normalised
    to lie in the unit interval so the exploration constant means the
    same thing whatever the test set is.
    """
    a = len(action_space)
    full = float(sum(len(t) for t in targets))
    root = ()
    N = {}
    W = {}
    best = []
    best_s, best_c = _score([], inputs, targets, n_reg, latency_weight,
                            reward_fn)
    for _ in range(int(n_sim)):
        node = root
        path = []
        while node in N and len(node) < int(max_len):
            tot = 0
            for k in range(a):
                tot += N[node][k]
            sq = math.sqrt(float(tot))
            bi = 0
            bv = None
            for k in range(a):
                nk = N[node][k]
                q = (W[node][k] / nk) if nk > 0 else 0.0
                u = q + c_puct * (1.0 / a) * sq / (1.0 + nk)
                if bv is None or u > bv:
                    bv, bi = u, k
            path.append((node, bi))
            node = node + (bi,)
        if node not in N:
            N[node] = [0] * a
            W[node] = [0.0] * a
        prog = [action_space[k] for k in node]
        s, c = _score(prog, inputs, targets, n_reg, latency_weight,
                      reward_fn)
        if s > best_s:
            best_s, best_c, best = s, c, prog
        v = (c / full) if full > 0 else 0.0
        for st, k in path:
            N[st][k] += 1
            W[st][k] += v
    return best, best_s, best_c, len(N)


def alphadev_quicksort_disc(target, action_space=None, reward_fn=None,
                            n_reg=2, max_len=3, latency_weight=0.0,
                            search="mcts", n_sim=400, c_puct=1.25,
                            seed=0):
    """Search for a machine program that sorts the given inputs.

    Parameters
    ----------
    target : sequence of sequences
        The test inputs. What each one should become is its own values
        in ascending order -- that is the specification of a sort, and
        making it derived rather than supplied means the two cannot
        drift apart.
    action_space : sequence or None
        The instructions the search may use. None builds every legal
        instruction over the memory and registers.
    reward_fn : callable or None
        ``f(program, inputs, targets, n_reg)`` returning the
        correctness. None is the count of correctly placed elements.
    n_reg : int
        How many registers the machine has.
    max_len : int
        The longest program the search will consider.
    latency_weight : float
        Instructions charged against correctness. Zero searches for
        correctness alone.
    search : {"mcts", "bfs"}
        Tree search, or exhaustive enumeration -- see the module
        docstring on why both are here.
    n_sim, c_puct : int, float
        Simulations and the PUCT exploration constant.

    Returns
    -------
    RichResult
        The best program, its score, its correctness, and how much of
        the space was looked at.

    References
    ----------
    Mankowitz et al. (2023) Nature 618, 257-263; Silver et al. (2017)
    Nature 550, 354-359.
    """
    inputs = [[float(v) for v in x] for x in target]
    if not inputs:
        raise ValueError("a search with no test input cannot tell a "
                         "sorting routine from any other program")
    n_mem = len(inputs[0])
    for x in inputs:
        if len(x) != n_mem:
            raise ValueError("every test input must be the same length")
    targets = [sorted(x) for x in inputs]
    n_reg = int(n_reg)
    if action_space is None:
        action_space = sort_action_space(n_mem, n_reg)
    action_space = [tuple(a) for a in action_space]
    if not action_space:
        raise ValueError("a search with no legal move has nothing to do")
    if search == "bfs":
        prog, s, c, seen = _bfs(inputs, targets, action_space, n_reg,
                                max_len, latency_weight, reward_fn)
    elif search == "mcts":
        prog, s, c, seen = _mcts(inputs, targets, action_space, n_reg,
                                 max_len, latency_weight, reward_fn,
                                 n_sim, c_puct, seed)
    else:
        raise ValueError("the search is mcts or bfs")
    full = sum(len(t) for t in targets)
    outs = [assembly_run(prog, x, n_reg) for x in inputs]
    return RichResult(payload={
        "program": [list(i) for i in prog],
        "text": program_text(prog),
        "length": len(prog),
        "score": s,
        "correct": c,
        "max_correct": full,
        "solved": c == full,
        "outputs": outs,
        "targets": targets,
        "nodes": seen,
        "n_actions": len(action_space),
        "n_mem": n_mem,
        "n_reg": n_reg,
        "max_len": int(max_len),
        "latency_weight": float(latency_weight),
        "search": search,
        "method": "AlphaDev AssemblyGame instruction search",
    })


def cheatsheet():
    return ("alfqud: AlphaDev AssemblyGame. Programs of mov/cmp/cmovl/"
            "cmovg searched by PUCT tree search or exhaustively, scored "
            "by executing them on test inputs")
