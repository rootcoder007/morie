"""Tests for egcn: the public e_gcn name for the EGNN of egnnL.

egcn re-exports egnnL's implementation; its own contract is that the
name reaches that implementation, and the equivariance that defines it
is checked once here through the egcn name (fully in test_egnnL)."""

import math

import pytest

import morie.fn.egnnL as egnnL
from morie.fn.egcn import e_gcn, egcn, run_egnn


def test_egcn_basic():
    assert e_gcn is egnnL.run_egnn and egcn is e_gcn and run_egnn is e_gcn


def test_egcn_edge():
    def phi_e(hi, hj, d2, a):
        return [math.tanh(hi[0] + d2)]

    def phi_x(m):
        return 0.1 * m[0]

    def phi_h(h, m):
        return [h[0] + m[0]]

    H = [[0.2], [0.5], [-0.1]]
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]
    th = 0.7
    Q = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]

    def move(P):
        return [[Q[0][0] * p[0] + Q[0][1] * p[1] + 4.0, Q[1][0] * p[0] + Q[1][1] * p[1] - 1.0] for p in P]

    a = e_gcn(H, X, 2, phi_e, phi_x, phi_h)
    b = e_gcn(H, move(X), 2, phi_e, phi_x, phi_h)
    for i in range(3):
        assert b["X"][i] == pytest.approx(move(a["X"])[i], abs=1e-10)
        assert b["H"][i] == pytest.approx(a["H"][i], abs=1e-10)
