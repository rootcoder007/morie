"""Tests for vitscn (Swin V2 scaled cosine attention, Sec 3.2)."""

import math

from morie.fn.vitscn import vit_scaled_cosine, vitscn

Q = [[1.0, 0.0], [0.6, 0.8]]
K = [[0.0, 1.0], [0.8, 0.6]]
V = [[1.0], [2.0]]


def test_vitscn_cosine_anchor():
    # rows are unit vectors, so with tau = 1 and B = 0 the pre-softmax
    # similarity IS the plain cosine: cos(q0,k0)=0, cos(q0,k1)=0.8,
    # cos(q1,k0)=0.8, cos(q1,k1)=0.96 -- hand-computed.
    r = vitscn(Q, K, V, tau=1.0)
    S = r["similarities"]
    assert abs(S[0][0] - 0.0) < 1e-12
    assert abs(S[0][1] - 0.8) < 1e-12
    assert abs(S[1][0] - 0.8) < 1e-12
    assert abs(S[1][1] - 0.96) < 1e-12
    # softmax of hand values
    w01 = math.exp(0.8) / (1.0 + math.exp(0.8))
    assert abs(r["weights"][0][1] - w01) < 1e-12
    assert abs(r["output"][0][0] - (1.0 * (1 - w01) + 2.0 * w01)) < 1e-12


def test_vitscn_tau_and_bias():
    # halving tau doubles the similarities (before bias)
    r1 = vitscn(Q, K, V, tau=1.0)
    r2 = vitscn(Q, K, V, tau=0.5)
    for a, b in zip(r1["similarities"], r2["similarities"]):
        assert all(abs(2 * x - y) < 1e-12 for x, y in zip(a, b))
    # a huge bias on column 0 concentrates every query on key 0
    r3 = vitscn(Q, K, V, tau=1.0, B=[[50.0, 0.0], [50.0, 0.0]])
    assert r3["weights"][0][0] > 1.0 - 1e-15
    assert abs(r3["output"][0][0] - 1.0) < 1e-12
    # paper constraint: tau must exceed 0.01
    try:
        vitscn(Q, K, V, tau=0.005)
        raise AssertionError("tau <= 0.01 must be rejected")
    except ValueError:
        pass
    assert vit_scaled_cosine is vitscn
