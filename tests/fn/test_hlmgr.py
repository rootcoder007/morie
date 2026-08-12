"""Tests for hlmgr (HLM random-effects covariance, Raudenbush & Bryk).

Replaces the generated stub, which imported ``hlm_gamma_matrix``.
"""

from morie.fn.hlmgr import hlmgr


def _betas(spread=5.0, n=12):
    # two coefficients per group, varying across groups
    out = []
    for j in range(n):
        out.append([spread * ((j % 4) - 1.5), 0.5 * ((j % 3) - 1)])
    return out


def test_tau_is_symmetric_and_has_a_non_negative_diagonal():
    res = hlmgr(_betas())
    tau = res["tau"]
    assert len(tau) == 2 and len(tau[0]) == 2
    assert abs(tau[0][1] - tau[1][0]) < 1e-12
    assert tau[0][0] >= 0 and tau[1][1] >= 0


def test_the_gamma_is_the_mean_of_the_group_coefficients():
    b = _betas()
    res = hlmgr(b)
    for k in range(2):
        want = sum(row[k] for row in b) / len(b)
        assert abs(res["gamma"][k] - want) < 1e-9


def test_the_total_spread_exceeds_the_between_group_part():
    b = _betas()
    V = [[[1.0, 0.0], [0.0, 1.0]] for _ in b]
    res = hlmgr(b, V)
    # T = S_total - mean(V), so the total must dominate
    assert res["s_total"][0][0] >= res["tau"][0][0] - 1e-12


def test_more_between_group_spread_gives_a_larger_tau():
    small = hlmgr(_betas(spread=0.5))["tau"][0][0]
    large = hlmgr(_betas(spread=20.0))["tau"][0][0]
    assert large > small


def test_reliabilities_lie_in_the_unit_interval():
    # one reliability matrix per group, lambda_j = T (T + V_j)^-1
    b = _betas()
    V = [[[1.0, 0.0], [0.0, 1.0]] for _ in b]
    res = hlmgr(b, V)
    assert len(res["reliabilities"]) == len(b)
    for mat in res["reliabilities"]:
        for k in range(2):
            assert -1e-9 <= mat[k][k] <= 1.0 + 1e-9


def test_shrunken_estimates_sit_between_the_group_and_the_mean():
    b = _betas()
    V = [[[4.0, 0.0], [0.0, 4.0]] for _ in b]
    res = hlmgr(b, V)
    for j in range(len(b)):
        for k in range(2):
            lo = min(b[j][k], res["gamma"][k])
            hi = max(b[j][k], res["gamma"][k])
            assert lo - 1e-9 <= res["shrunken"][j][k] <= hi + 1e-9


def test_validation():
    b = _betas()
    for call in (lambda: hlmgr(b[:2]),
                 lambda: hlmgr([[1.0, 2.0], [1.0], [1.0, 2.0]]),
                 lambda: hlmgr(b, [[[1.0, 0.0], [0.0, 1.0]]])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
