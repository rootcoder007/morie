"""Tests for kmcrag.kamath_corrective_rag."""

from morie.fn import _array_core as np

from morie.fn.kmcrag import kamath_corrective_rag


def test_kmcrag_basic():
    """Test basic functionality."""
    query = 'q'
    docs = ['d1', 'd2']
    clf = lambda q, d: 0.05
    tau_hi = 0.8
    tau_lo = 0.2
    result = kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcrag_edge():
    """Test edge cases."""
    query = 'q'
    docs = ['d1', 'd2']
    clf = lambda q, d: 0.05
    tau_hi = 0.8
    tau_lo = 0.2
    result = kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo)
    assert isinstance(result, dict)
