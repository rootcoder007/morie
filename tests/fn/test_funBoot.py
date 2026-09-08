"""Tests for funBoot (functional bootstrap bands).

Replaces the generated stub, which imported ``functional_bootstrap``.
"""

import math

from morie.fn.funBoot import funBoot


def _curves(n=40, m=25, seed=3, noise=0.2):
    st = [seed]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    out = []
    for _ in range(n):
        shift = noise * (r() - 0.5)
        out.append([math.sin(2 * math.pi * t / m) + shift
                    for t in range(m)])
    return out


def test_the_centre_tracks_the_pointwise_mean():
    # the centre is the bootstrap estimate, so it sits near the sample
    # mean curve rather than exactly on it
    curves = _curves()
    res = funBoot(curves, B=50, seed=1)
    assert len(res["center"]) == len(curves[0])
    for t in range(len(curves[0])):
        want = sum(c[t] for c in curves) / len(curves)
        assert abs(res["center"][t] - want) < 0.05


def test_the_band_covers_the_stated_share_of_curves():
    curves = _curves()
    res = funBoot(curves, alpha=0.05, B=200, seed=1)
    assert res["radius"] > 0
    # n_within counts the bootstrap replicates inside the band, so it is
    # bounded by B and should sit near the nominal 95%
    assert 0 <= res["n_within"] <= 200
    assert res["n_within"] / 200.0 > 0.85


def test_a_tighter_alpha_gives_a_wider_band():
    curves = _curves()
    wide = funBoot(curves, alpha=0.01, B=150, seed=1)["radius"]
    narrow = funBoot(curves, alpha=0.25, B=150, seed=1)["radius"]
    assert wide > narrow


def test_noisier_curves_need_a_wider_band():
    tight = funBoot(_curves(noise=0.05), B=150, seed=1)["radius"]
    loose = funBoot(_curves(noise=2.0), B=150, seed=1)["radius"]
    assert loose > tight


def test_both_metrics_run():
    curves = _curves()
    for metric in ("l2", "sup"):
        res = funBoot(curves, metric=metric, B=80, seed=1)
        assert res["metric"] == metric
        assert res["radius"] > 0


def test_seed_reproducibility():
    curves = _curves()
    a = funBoot(curves, B=60, seed=5)["radius"]
    b = funBoot(curves, B=60, seed=5)["radius"]
    assert a == b


def test_validation():
    curves = _curves(n=10, m=5)
    for call in (lambda: funBoot(curves[:2]),
                 lambda: funBoot([[1.0, 2.0], [1.0], [1.0, 2.0]]),
                 lambda: funBoot(curves, alpha=0.0),
                 lambda: funBoot(curves, metric="l1")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
