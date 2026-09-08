"""Anchored tests for chipsq (Zhang et al. 2008 MACS lambda_local).

Anchors: closed-form Poisson upper tail
P(X >= k) = 1 - e^{-lam} sum_{i<k} lam^i / i!, and the hand-checked
lambda_local = max(lambda_BG, lambda_1k, lambda_5k, lambda_10k) with
window counts rescaled to the peak width.
"""

import math

from morie.fn.chipsq import chipsq, chip_seq_peak


def _pois_upper_closed(k, lam):
    return 1.0 - math.exp(-lam) * sum(lam ** i / math.factorial(i)
                                      for i in range(k))


def test_chipsq_lambda_local_hand():
    # width 500: lam1 = 10*500/1000 = 5, lam5 = 30*500/5000 = 3,
    # lam10 = 80*500/10000 = 4, bg = 2 -> lambda_local = 5.
    res = chipsq(12, 500.0, 2.0, count_1k=10, count_5k=30, count_10k=80)
    assert abs(res["lambda_local"][0] - 5.0) < 1e-15
    assert abs(res["fold_enrichment"][0] - 12.0 / 5.0) < 1e-15


def test_chipsq_no_control_drops_1k():
    res = chipsq(12, 500.0, 2.0, count_1k=10, count_5k=30, count_10k=80,
                 use_1k=False)
    assert abs(res["lambda_local"][0] - 4.0) < 1e-15


def test_chipsq_poisson_closed_form():
    for k, lam in [(3, 1.234), (12, 5.0), (1, 0.7), (25, 10.5)]:
        res = chipsq(k, 1000.0, lam)
        assert abs(res["pvalue"][0] - _pois_upper_closed(k, lam)) < 1e-10


def test_chipsq_vector_and_edges():
    res = chipsq([0, 5], [500.0, 500.0], 2.0)
    assert res["pvalue"][0] == 1.0  # k = 0
    assert res["n_peaks"] == 2
    assert chip_seq_peak is chipsq
