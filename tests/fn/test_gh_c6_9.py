"""Tests for gh_c6_9.ghosal_kl_perm."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gh_c6_9 import ghosal_kl_perm


def test_gh_c6_9_basic():
    """Test basic functionality: product additivity on small arrays."""
    p0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q0 = np.array([2.0, 2.0, 2.0])
    p = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
    q = np.array([1.0, 3.0, 5.0])

    result = ghosal_kl_perm(p0, q0, p, q)

    # The function returns a RichResult-like object with these keys.
    assert "estimate" in result
    assert "kl_marginals" in result
    assert "additivity_gap" in result

    est = float(np.asarray(result["estimate"], dtype=float))
    km = result["kl_marginals"]
    gap = float(np.asarray(result["additivity_gap"], dtype=float))

    assert np.all(np.isfinite(est))
    assert np.all(np.isfinite(km))

    # Independent reference: KL additivity for product distributions.
    def kl(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        sa = float(np.sum(a))
        sb = float(np.sum(b))
        a = a / sa
        b = b / sb
        s = 0.0
        for i in range(len(a)):
            if a[i] > 0:
                denom = b[i] if b[i] > 1e-300 else 1e-300
                s += float(a[i]) * (float(np.log(a[i])) - float(np.log(denom)))
        return s

    k1_ref = kl(p0, p)
    k2_ref = kl(q0, q)

    p0q0_ref = []
    for x in np.asarray(p0, dtype=float) / float(np.sum(np.asarray(p0, dtype=float))):
        for y in np.asarray(q0, dtype=float) / float(np.sum(np.asarray(q0, dtype=float))):
            p0q0_ref.append(float(x) * float(y))
    pq_ref = []
    for x in np.asarray(p, dtype=float) / float(np.sum(np.asarray(p, dtype=float))):
        for y in np.asarray(q, dtype=float) / float(np.sum(np.asarray(q, dtype=float))):
            pq_ref.append(float(x) * float(y))
    kprod_ref = kl(p0q0_ref, pq_ref)

    # Additivity gap is essentially zero.
    assert abs(gap) < 1e-9

    # Marginals match the reference KL computations.
    assert abs(float(km[0]) - k1_ref) < 1e-9
    assert abs(float(km[1]) - k2_ref) < 1e-9

    # Product KL matches the independently computed reference.
    assert abs(est - kprod_ref) < 1e-9


def test_gh_c6_9_edge():
    """Test edge cases: minimal non-trivial support for both marginals."""
    p0 = np.array([42.0])
    q0 = np.array([1.0])
    p = np.array([7.0])
    q = np.array([3.0])

    result = ghosal_kl_perm(p0, q0, p, q)

    # Documented behaviour: return keys are estimate / kl_marginals /
    # additivity_gap / method. There is no 'n' key.
    assert "estimate" in result
    assert "kl_marginals" in result
    assert "additivity_gap" in result
    assert "method" in result

    # With a single element on each side, KL divergence is zero and
    # both the product estimate and the additivity gap vanish.
    assert abs(float(np.asarray(result["estimate"], dtype=float))) < 1e-12
    assert abs(float(np.asarray(result["kl_marginals"][0], dtype=float))) < 1e-12
    assert abs(float(np.asarray(result["kl_marginals"][1], dtype=float))) < 1e-12
    assert abs(float(np.asarray(result["additivity_gap"], dtype=float))) < 1e-12
