"""ptqdr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptqdr import quadrat_test


def test_ptqdr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        quadrat_test(data=None)
