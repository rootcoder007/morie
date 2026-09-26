"""Hommel adjusted p-values against R p.adjust(method = "hommel")."""

from morie.multiple_testing import hommel

CASES = [
    ([0.01, 0.02, 0.03, 0.04, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05]),
    ([0.001, 0.2, 0.03, 0.5, 0.049, 0.011], [0.006, 0.4, 0.098, 0.5, 0.147, 0.055]),
    ([0.04, 0.04, 0.3], [0.08, 0.08, 0.3]),
]


def test_hommel_matches_p_adjust():
    for p, ref in CASES:
        got = [float(v) for v in hommel(p).adjusted]
        for a, b in zip(got, ref):
            assert abs(a - b) <= 1e-12
