"""Majority tournaments, the yolk, amendment agendas and agenda setting: brute force and theorems."""

import itertools
import math

from morie.fn._rng import random_uniform
from morie.fn.svagnd import amendment_agenda
from morie.fn.svgeom import yolk
from morie.fn.svsetr import agenda_setter_equilibrium
from morie.fn.svtour import majority_tournament


def brute(beats, m):
    banks = set()
    for r in range(1, m + 1):
        for S in itertools.combinations(range(m), r):
            order = sorted(S, key=lambda a: -sum(beats[a][b] for b in S))
            if all(beats[order[i]][order[j]] for i in range(r) for j in range(i + 1, r)) and not any(
                all(beats[c][x] for x in S) for c in range(m) if c not in S
            ):
                banks.add(order[0])
    unc = [
        b
        for b in range(m)
        if not any(beats[a][b] and all(beats[a][c] for c in range(m) if beats[b][c]) for a in range(m))
    ]
    for r in range(1, m + 1):
        for S in itertools.combinations(range(m), r):
            if all(beats[a][b] for a in S for b in range(m) if b not in S):
                return sorted(banks), unc, list(S)


def test_tournament_solutions_match_brute_force():
    U = [float(u) for u in random_uniform(4000, seed=5, stream=0)]
    cyclic = 0
    for t in range(60):
        nv, na = 7 + 2 * (t % 3), 5 + t % 3
        V = [[U[40 * t + 2 * i], U[40 * t + 2 * i + 1]] for i in range(nv)]
        A = [[U[2000 + 20 * t + 2 * j], U[2001 + 20 * t + 2 * j]] for j in range(na)]
        r = majority_tournament(V, A)
        banks, unc, smith = brute(r["beats"], na)
        assert r["banks"] == banks and r["uncovered"] == unc and r["top_cycle"] == smith
        assert set(r["banks"]) <= set(r["uncovered"]) <= set(r["top_cycle"])  # Banks in UC in TC
        cyclic += r["cyclic"]
        if r["condorcet_winner"] is not None:
            assert r["top_cycle"] == [r["condorcet_winner"]] and r["copeland"][r["condorcet_winner"]] == na - 1
    assert cyclic > 0


def test_condorcet_loser_copeland_borda():
    r = majority_tournament([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]])
    assert (
        r["condorcet_winner"] == 1 and r["condorcet_loser"] is None and r["copeland"] == [-1, 2, -1]
    )  # alternatives 0 and 2 tie (the middle voter is indifferent)
    r = majority_tournament([[0.0], [0.1], [3.0]], [[0.0], [1.0], [4.0]])
    assert r["condorcet_winner"] == 0 and r["condorcet_loser"] == 2 and r["copeland"] == [2, 0, -2]
    assert r["borda"] == [2 + 2 + 0, 1 + 1 + 1, 0 + 0 + 2]


def test_yolk_properties():
    r = yolk([(0, 0), (1, 1), (-1, -1), (2, -1), (-2, 1)])
    assert r["core"] and max(abs(v) for v in r["center"]) < 1e-6  # Plott radial symmetry about the origin
    assert not yolk([(0, 0), (1, 1), (-1, -1), (2, -1), (-2, 1.3)])["core"]
    U = [float(u) for u in random_uniform(4000, seed=8, stream=0)]
    P = [(U[2 * i], U[2 * i + 1]) for i in range(7)]
    y = yolk(P)
    assert abs(y["radius"] - 0.041686079839) < 1e-9  # R arm Yolk: 0.041686079839
    dirs = [(math.cos(math.pi * k / 4000), math.sin(math.pi * k / 4000)) for k in range(4000)]
    for u in dirs[::7]:  # every median line comes within the radius of the centre
        med = sorted(p[0] * u[0] + p[1] * u[1] for p in P)[3]
        assert abs(y["center"][0] * u[0] + y["center"][1] * u[1] - med) <= y["radius"] + 1e-9


def test_agendas_and_setter():
    U = [float(u) for u in random_uniform(4000, seed=8, stream=0)]
    for t in range(40):
        V = [[U[1000 + 20 * t + 2 * i], U[1001 + 20 * t + 2 * i]] for i in range(7)]
        G = [[U[2500 + 12 * t + 2 * j], U[2501 + 12 * t + 2 * j]] for j in range(5)]
        a = amendment_agenda(V, G)
        assert a["sophisticated"] in majority_tournament(V, G)["uncovered"]  # Shepsle and Weingast (1984)
    a = amendment_agenda([[0.0], [0.4], [1.0]], [[1.0], [0.0], [0.5]])
    assert a["sincere_path"] == [0, 1, 2] and a["sincere"] == 2
    assert agenda_setter_equilibrium(10.0, 4.0, 1.0)["outcome"] == 7.0  # acceptance set [1, 7]
    assert agenda_setter_equilibrium(5.0, 4.0, 1.0)["outcome"] == 5.0
    assert agenda_setter_equilibrium(10.0, 4.0, 1.0, options=[2, 6.5, 8])["outcome"] == 6.5
    assert agenda_setter_equilibrium(10.0, 4.0, 1.0, options=[9, 12])["outcome"] == 1.0


def test_power_indices_brute_force():
    from morie.fn.svpowr import power_indices

    w, q = [4, 3, 2, 1], 6
    r = power_indices(w, q)
    ss = [0] * 4
    for perm in itertools.permutations(range(4)):
        s = 0
        for i in perm:
            s += w[i]
            if s >= q:
                ss[i] += 1
                break
    assert all(abs(a - b / 24) < 1e-15 for a, b in zip(r["shapley_shubik"], ss))
    assert r["minimal_winning"] == [[0, 1], [0, 2], [1, 2, 3]] and r["min_winning_size"] == 2
    assert all(abs(a - b) < 1e-15 for a, b in zip(r["deegan_packel"], [1 / 3, 5 / 18, 5 / 18, 1 / 9]))
    assert all(abs(a - b) < 1e-15 for a, b in zip(r["holler"], [2 / 7, 2 / 7, 2 / 7, 1 / 7]))
    assert all(abs(a - b) < 1e-15 for a, b in zip(r["banzhaf"], [5 / 12, 3 / 12, 3 / 12, 1 / 12]))
    assert abs(sum(r["johnston"]) - 1) < 1e-15 and r["banzhaf_absolute"][0] == 5 / 8
    assert power_indices([1, 1, 1])["banzhaf"] == [1 / 3] * 3  # default quota: simple majority


def test_shapley_owen_matches_direction_sampling():
    from morie.fn.svsowen import shapley_owen

    U = [float(u) for u in random_uniform(20, seed=2, stream=0)]
    P = [(U[2 * i], U[2 * i + 1]) for i in range(7)]
    so = shapley_owen(P)["value"]
    assert abs(sum(so) - 1) < 1e-15
    cnt = [0] * 7
    for k in range(20000):
        th = math.pi * (k + 0.5) / 20000
        cnt[sorted(range(7), key=lambda i: P[i][0] * math.cos(th) + P[i][1] * math.sin(th))[3]] += 1
    assert all(abs(a - c / 20000) < 2e-4 for a, c in zip(so, cnt))
    assert shapley_owen([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])["value"] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_top_cycle_with_ties_between_members():
    # even electorates create majority ties; the Smith set must absorb anything tied with a member
    U = [float(u) for u in random_uniform(4000, seed=12, stream=0)]
    tied = 0
    for t in range(80):
        V = [[round(U[30 * t + 2 * i], 1), round(U[30 * t + 2 * i + 1], 1)] for i in range(4)]
        A = [[round(U[3000 + 10 * t + 2 * j], 1), round(U[3001 + 10 * t + 2 * j], 1)] for j in range(4)]
        r = majority_tournament(V, A)
        tied += any(r["margin"][a][b] == 0 for a in range(4) for b in range(4) if a != b)
        assert r["top_cycle"] == brute(r["beats"], 4)[2]
    assert tied > 20
