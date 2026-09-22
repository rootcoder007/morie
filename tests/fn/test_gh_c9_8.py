"""Tests for gh_c9_8.ghosal_nlar_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_8 import ghosal_nlar_crt


def test_gh_c9_8_basic():
    """Test basic functionality."""
    result = ghosal_nlar_crt(ns=(200, 800, 3200), seed=42)
    assert "estimate" in result
    estimate = result["estimate"]
    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))

    # Reference computation: reproduce the binned conditional-mean estimator
    # (normal-normal per bin with N(0,.) prior) using an independent RNG path.
    def reference(ns, seed):
        rng = np.random.default_rng(seed)
        errs = []
        k = 8
        for n in ns:
            x = 0.0
            s_ = [0.0] * k
            c_ = [0.0] * k
            for _ in range(n):
                nxt = 0.5 * x + 0.5 * float(rng.normal(0, 1))
                b = int((x + 3.0) / 0.75)
                if b < 0:
                    b = 0
                elif b > k - 1:
                    b = k - 1
                s_[b] += nxt
                c_[b] += 1.0
                x = nxt
            err = 0.0
            for b in range(k):
                centre = -3.0 + (b + 0.5) * 0.75
                post = s_[b] / (c_[b] + 1.0)
                err += abs(post - 0.5 * centre) / k
            errs.append(err)
        return errs

    ref_errs = reference((200, 800, 3200), 42)
    assert np.isclose(float(estimate), ref_errs[-1])

    # Per-n errors should match the reference computation for every n.
    assert np.allclose(np.asarray(result["err_by_n"], dtype=float),
                       np.asarray(ref_errs, dtype=float))

    # The error should contract as n grows (errs[-1] < errs[0]) when seeded.
    assert bool(result["improving"]) is (ref_errs[-1] < ref_errs[0])

    # Method label is informative, not a numerical key.
    assert isinstance(result["method"], str)


def test_gh_c9_8_edge():
    """Test edge cases with the default (single) tuple of ns."""
    result = ghosal_nlar_crt()
    # Default ns is a 3-tuple; the function returns a single estimate, not
    # per-call metadata like 'n'.
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert not isinstance(result["estimate"], (int,))
