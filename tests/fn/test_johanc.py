"""Tests for johanc.johansen_cointegration.

``johanc`` used to carry a pasted one-sample Kolmogorov-Smirnov test.
A correct Johansen reduced-rank test already lives in ``morie.fn.johsn``,
so ``johanc`` is now a re-export of it rather than a second copy.  These
tests pin that re-export and the basic shape of the result; the
statistical content is tested in ``test_johsn.py``.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn import johanc, johsn
from morie.fn.johanc import johansen_cointegration


def test_johanc_is_the_johsn_implementation():
    assert johanc.johansen_cointegration is johsn.johansen_cointegration


def test_johanc_has_no_ks_residue():
    src = open(johanc.__file__).read()
    for token in ("d_plus", "d_minus", "ecdf", "ksone"):
        assert token not in src, "pasted KS body still present: " + token


def test_johanc_runs_on_a_cointegrated_pair():
    n = 120
    common = np.array([float((i * 7) % 13) + 0.5 * i for i in range(n)])
    Y = np.column_stack([common + np.array([float((i * 3) % 5) for i in range(n)]),
                         2.0 * common + np.array([float((i * 11) % 7) for i in range(n)])])
    res = johansen_cointegration(Y, k_ar_diff=1)
    assert len(res["trace_stat"]) == 2
    assert len(res["eigenvalues"]) >= 2
    assert 0 <= res["rank"] <= 2


def test_johanc_rejects_a_univariate_series():
    with pytest.raises(ValueError):
        johansen_cointegration(np.arange(100.0))
