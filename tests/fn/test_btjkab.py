"""Tests for btjkab.boot_jackknife_after_boot."""

import numpy as np

from morie.fn.btjkab import boot_jackknife_after_boot


def test_btjkab_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    B = 100
    x = rng.normal(0, 1, n)
    theta_b = rng.normal(0, 1, B)
    # B_idx: B resamples, each a sequence of distinct 0-based indices into x.
    # Build it deterministically: resample b uses indices [b, b+1, ..., b+n-1] mod n
    # (with duplicates removed via a per-b round-robin pick of m = n//2 indices).
    m = n // 2  # 50 indices per resample
    B_idx = []
    for b in range(B):
        idx = [(b + k) % n for k in range(m)]
        B_idx.append(idx)

    result = boot_jackknife_after_boot(x, theta_b, B_idx)

    # The function returns a RichResult; support dict-like access for the test.
    assert hasattr(result, "payload")
    payload = result.payload

    # Required documented keys.
    assert "infl_i" in payload
    assert "theta_minus" in payload
    assert "n_out" in payload
    assert "grand_mean" in payload
    assert "max_abs_influence" in payload
    assert "estimate" in payload

    infl = payload["infl_i"]
    tm = payload["theta_minus"]
    nout = payload["n_out"]

    assert len(infl) == n
    assert len(tm) == n
    assert len(nout) == n

    # Independent recomputation from the documented formula.
    used = [[False] * n for _ in range(B)]
    for b in range(B):
        for i in B_idx[b]:
            used[b][i] = True

    tb = list(theta_b)
    grand_independent = sum(tb) / len(tb)

    infl_independent = []
    tm_independent = []
    nout_independent = []
    for i in range(n):
        s = 0.0
        c = 0
        for b in range(B):
            if not used[b][i]:
                s += tb[b]
                c += 1
        nout_independent.append(c)
        if c == 0:
            tm_independent.append(float("nan"))
            infl_independent.append(float("nan"))
        else:
            m_val = s / c
            tm_independent.append(m_val)
            infl_independent.append(m_val - grand_independent)

    max_abs_independent = max(
        (abs(v) for v in infl_independent if v == v), default=0.0
    )

    assert payload["grand_mean"] == grand_independent
    assert payload["max_abs_influence"] == max_abs_independent
    assert payload["estimate"] == max_abs_independent
    assert list(nout) == nout_independent
    assert list(tm) == tm_independent
    assert list(infl) == infl_independent


def test_btjkab_edge():
    """Test edge cases: every resample excludes at least one observation."""
    rng = np.random.default_rng(42)
    n = 20
    B = 10
    x = rng.normal(0, 1, n)
    theta_b = rng.normal(0, 1, B)

    # Each resample excludes exactly observation i=b (uses all others).
    # Then for observation i, n_out[i] == 1 and theta_minus[i] is the single
    # theta_b value from resample b=i.
    B_idx = [[j for j in range(n) if j != b] for b in range(B)]

    result = boot_jackknife_after_boot(x, theta_b, B_idx)
    payload = result.payload

    assert payload["n"] == n
    assert payload["B"] == B

    # For i in [0, B): exactly one resample omitted i -> n_out[i] == 1.
    # For i >= B: no resample omitted i -> n_out[i] == 0 -> NaN influence.
    for i in range(B):
        assert payload["n_out"][i] == 1
        assert payload["theta_minus"][i] == theta_b[i]
    for i in range(B, n):
        assert payload["n_out"][i] == 0
        assert payload["theta_minus"][i] != payload["theta_minus"][i]  # NaN
        assert payload["infl_i"][i] != payload["infl_i"][i]  # NaN
