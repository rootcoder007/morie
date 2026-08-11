"""Tests for yarn (Peng et al. 2023, Eqs 17/18/20/22)."""

import math

from morie.fn.yarn import yarn, yarn_context_scaling


def test_yarn_limiting_cases():
    # s = 1: nothing changes and the temperature is exactly 1.
    r = yarn(10000.0, 1.0, 8, 2048.0)
    assert r["theta_new"] == r["theta"]
    assert r["temperature"] == 1.0
    # tiny L: every dimension has r < beta_slow -> pure interpolation
    r = yarn(10000.0, 2.0, 8, 1e-6)
    assert all(abs(n - t / 2.0) < 1e-18 for n, t in zip(r["theta_new"], r["theta"]))
    # huge L: every dimension has r > beta_fast -> untouched
    r = yarn(10000.0, 2.0, 8, 1e9)
    assert r["theta_new"] == r["theta"]


def test_yarn_hand_anchor():
    # base 10000, d = 8, L = 2048, s = 2 -- hand-computed from the
    # paper equations: theta = (1, 10^-1, 10^-2, 10^-3);
    # r(d) = L theta / (2 pi) = (325.949.., 32.59.., 3.259.., 0.3259..);
    # gamma = (1, 1, (3.2595-1)/31, 0); Eq 20 blend; Eq 22 temperature.
    r = yarn(10000.0, 2.0, 8, 2048.0)
    th = [1.0, 10.0 ** -1, 10.0 ** -2, 10.0 ** -3]
    assert all(abs(a - b) < 1e-15 for a, b in zip(r["theta"], th))
    rot2 = 2048.0 * 0.01 / (2.0 * math.pi)
    g2 = (rot2 - 1.0) / 31.0
    want2 = (1.0 - g2) * 0.01 / 2.0 + g2 * 0.01
    assert abs(r["gamma"][2] - g2) < 1e-15
    assert abs(r["theta_new"][2] - want2) < 1e-15
    assert r["theta_new"][0] == 1.0 and r["theta_new"][1] == 0.1
    assert abs(r["theta_new"][3] - 0.0005) < 1e-18
    inv_t = (0.1 * math.log(2.0) + 1.0) ** 2
    assert abs(r["logit_scale"] - inv_t) < 1e-15
    assert abs(r["temperature"] - 1.0 / inv_t) < 1e-15


def test_yarn_wrapper():
    a = yarn_context_scaling(theta=[1.0, 0.1, 0.01, 0.001], s=2.0, d=8, L=2048.0)
    b = yarn(10000.0, 2.0, 8, 2048.0)
    assert all(abs(x - y) < 1e-15 for x, y in zip(a["theta_new"], b["theta_new"]))
