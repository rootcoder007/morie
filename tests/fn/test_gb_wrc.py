"""Tests for gb_wrc.gibbons_runs_critical."""

from morie.fn import _array_core as np

from morie.fn.gb_wrc import gibbons_runs_critical


def _runs_pmf(n1, n2):
    """Independent reference for the total-runs PMF (Gibbons & Chakraborti, Sec. 3.2)."""
    from math import comb
    n = n1 + n2
    den = comb(n, n1)
    support = list(range(2, n + 1))
    pmf = []
    for rr in support:
        if rr % 2 == 0:
            k = rr // 2
            p = 2.0 * comb(n1 - 1, k - 1) * comb(n2 - 1, k - 1)
        else:
            k = (rr - 1) // 2
            p = (
                comb(n1 - 1, k - 1) * comb(n2 - 1, k)
                + comb(n1 - 1, k) * comb(n2 - 1, k - 1)
            )
        pmf.append(p / den)
    return support, pmf


def test_gb_wrc_basic():
    """Two-sided critical region matches the largest exact-level region."""
    n1, n2 = 7, 9
    alpha = 0.05
    result = gibbons_runs_critical(n1, n2, alpha, tail="two-sided")

    # Returned object is a RichResult; documented keys must be present.
    assert hasattr(result, "keys") or isinstance(result, dict)
    payload = result.payload if hasattr(result, "payload") else result

    for key in ("lower", "upper", "alpha_lower", "alpha_upper",
                "alpha_exact", "n1", "n2", "method"):
        assert key in payload, f"missing key {key!r}"

    # Round-trip of n1/n2.
    assert int(payload["n1"]) == n1
    assert int(payload["n2"]) == n2

    # Independently reproduce the documented rule: largest c with P(R<=c) <= a,
    # smallest c with P(R>=c) <= a, where a = alpha/2 for two-sided.
    support, pmf = _runs_pmf(n1, n2)
    a = alpha / 2.0

    expected_lower = float("nan")
    expected_al = 0.0
    acc = 0.0
    for s, p in zip(support, pmf):
        acc += p
        if acc <= a:
            expected_lower = float(s)
            expected_al = acc
        else:
            break

    expected_upper = float("nan")
    expected_au = 0.0
    acc = 0.0
    for s, p in zip(reversed(support), reversed(pmf)):
        acc += p
        if acc <= a:
            expected_upper = float(s)
            expected_au = acc
        else:
            break

    assert payload["lower"] == expected_lower
    assert payload["upper"] == expected_upper
    assert abs(payload["alpha_lower"] - expected_al) < 1e-12
    assert abs(payload["alpha_upper"] - expected_au) < 1e-12
    assert abs(payload["alpha_exact"] - (expected_al + expected_au)) < 1e-12
    assert payload["alpha_exact"] <= alpha
    assert payload["method"].startswith("exact runs-test")


def test_gb_wrc_edge():
    """One-sided tails return only the relevant critical value."""
    n1, n2 = 5, 6
    alpha = 0.05

    # Left tail: clustering alternative -> only a lower critical value.
    res_l = gibbons_runs_critical(n1, n2, alpha, tail="left")
    payload_l = res_l.payload if hasattr(res_l, "payload") else res_l
    support, pmf = _runs_pmf(n1, n2)
    a = alpha

    expected_lower = float("nan")
    expected_al = 0.0
    acc = 0.0
    for s, p in zip(support, pmf):
        acc += p
        if acc <= a:
            expected_lower = float(s)
            expected_al = acc
        else:
            break

    assert payload_l["lower"] == expected_lower
    assert abs(payload_l["alpha_lower"] - expected_al) < 1e-12
    # No right-tail rejection region requested.
    assert payload_l["upper"] != payload_l["upper"]  # NaN
    assert payload_l["alpha_upper"] == 0.0
    assert payload_l["alpha_exact"] == expected_al

    # Right tail: mixing alternative -> only an upper critical value.
    res_r = gibbons_runs_critical(n1, n2, alpha, tail="right")
    payload_r = res_r.payload if hasattr(res_r, "payload") else res_r

    expected_upper = float("nan")
    expected_au = 0.0
    acc = 0.0
    for s, p in zip(reversed(support), reversed(pmf)):
        acc += p
        if acc <= a:
            expected_upper = float(s)
            expected_au = acc
        else:
            break

    assert payload_r["upper"] == expected_upper
    assert abs(payload_r["alpha_upper"] - expected_au) < 1e-12
    assert payload_r["lower"] != payload_r["lower"]  # NaN
    assert payload_r["alpha_lower"] == 0.0
    assert payload_r["alpha_exact"] == expected_au
