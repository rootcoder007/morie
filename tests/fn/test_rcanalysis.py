"""Tests for rcanalysis: roll-call cohesion, party influence and logistic ideal points."""

import math

from morie.fn.rcanalysis import logit_ideal_points, party_influence, rollcall_cohesion

N, M = 20, 30
X = [math.sin(i * 1.3) * 1.5 + (0.8 if i % 2 else -0.8) for i in range(N)]
V = [
    [1 if math.sin(j * 0.7) + (0.5 + (0.3 * j) % 5) * X[i] + 0.3 * math.cos(i * j) > 0 else 0 for j in range(M)]
    for i in range(N)
]
V[2][3] = None
PARTY = ["L" if i % 2 else "R" for i in range(N)]


def test_cohesion_indices_by_hand():
    r = rollcall_cohesion(V, PARTY)
    rows = [i for i in range(N) if PARTY[i] == "L"]
    for j in range(M):
        y = sum(1 for i in rows if V[i][j] == 1)
        no = sum(1 for i in rows if V[i][j] == 0)
        assert r.rice["L"][j] == abs(y - no) / (y + no)
    assert r.participation[2] == (M - 1) / M
    assert r.pairwise[0][0] == 1.0


def test_party_influence_regression():
    pd = [1 if i % 2 else 0 for i in range(N)]
    r = party_influence(V, pd, lopsided=0.7)
    for j, g in zip(r.close_votes, r.party_effect):
        rows = [i for i in range(N) if V[i][j] is not None]
        # the normal equations hold: residuals orthogonal to the party dummy
        # (solve the 3 x 3 system independently)
        X = [[1.0, r.ideal_points[i], pd[i]] for i in rows]
        y = [float(V[i][j]) for i in rows]
        G = [[sum(a[p] * a[q] for a in X) for q in range(3)] for p in range(3)]
        h = [sum(a[p] * v for a, v in zip(X, y)) for p in range(3)]
        det = (
            G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
            - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
            + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0])
        )
        num = (
            G[0][0] * (G[1][1] * h[2] - h[1] * G[2][1])
            - G[0][1] * (G[1][0] * h[2] - h[1] * G[2][0])
            + h[0] * (G[1][0] * G[2][1] - G[1][1] * G[2][0])
        )
        assert abs(g - num / det) <= 1e-9


def test_logit_ideal_points_identification_and_recovery():
    r = logit_ideal_points(V, prior_sd=3.0)
    x = r.ideal_points
    assert abs(sum(x)) <= 1e-9 and abs(sum(v * v for v in x) / N - 1) <= 1e-9
    mx = sum(X) / N
    cor = sum((a - mx) * b for a, b in zip(X, x)) / math.sqrt(sum((a - mx) ** 2 for a in X) * N)
    assert abs(cor) > 0.9
    assert r.iterations < 500
