"""Tests for mctsr (Monte Carlo tree search with UCT).

Replaces the generated stub, which imported ``mcts_rollout``.
"""

from morie.fn.mctsr import mctsr


# A one-move game: from the root, action "good" reaches a state worth 1
# and action "bad" a state worth 0. Search must pick "good".
def _one_move_game():
    def actions(s):
        return [] if s in ("good", "bad") else ["good", "bad"]

    def step(s, a):
        return a

    def reward(s):
        return 1.0 if s == "good" else 0.0

    def is_terminal(s):
        return s in ("good", "bad")

    return actions, step, reward, is_terminal


def test_it_finds_the_better_of_two_moves():
    a, s, r, t = _one_move_game()
    res = mctsr("root", a, s, r, t, n_iter=100, seed=1)
    assert res["action"] == "good"
    assert res["root_visits"] == 100
    assert res["child_visits"]["good"] > res["child_visits"]["bad"]


def test_the_reported_values_match_the_rewards():
    a, s, r, t = _one_move_game()
    res = mctsr("root", a, s, r, t, n_iter=80, seed=1)
    # child_values holds the mean return of each child, not the sum
    assert abs(res["child_values"]["good"] - 1.0) < 1e-9
    assert abs(res["child_values"]["bad"]) < 1e-9
    assert sum(res["child_visits"].values()) == res["root_visits"]


def test_more_iterations_concentrate_the_visits():
    a, s, r, t = _one_move_game()
    few = mctsr("root", a, s, r, t, n_iter=40, seed=1)
    many = mctsr("root", a, s, r, t, n_iter=400, seed=1)
    assert (many["child_visits"]["good"] / 400.0) > \
        (few["child_visits"]["good"] / 40.0)


def test_exploration_constant_zero_is_greedy():
    a, s, r, t = _one_move_game()
    res = mctsr("root", a, s, r, t, n_iter=60, c=0.0, seed=1)
    assert res["child_visits"]["good"] > res["child_visits"]["bad"]


def test_both_final_move_rules_and_both_backups_run():
    a, s, r, t = _one_move_game()
    for final in ("robust", "max"):
        assert mctsr("root", a, s, r, t, n_iter=60, seed=1,
                     final=final)["action"] == "good"
    for backup in ("sum", "negamax"):
        assert mctsr("root", a, s, r, t, n_iter=60, seed=1,
                     backup=backup)["backup"] == backup


def test_seed_reproducibility():
    a, s, r, t = _one_move_game()
    x = mctsr("root", a, s, r, t, n_iter=50, seed=9)["child_visits"]
    y = mctsr("root", a, s, r, t, n_iter=50, seed=9)["child_visits"]
    assert x == y


def test_validation():
    a, s, r, t = _one_move_game()
    for call in (lambda: mctsr("root", a, s, r, t, backup="mean"),
                 lambda: mctsr("root", a, s, r, t, final="first"),
                 lambda: mctsr("good", a, s, r, t)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
