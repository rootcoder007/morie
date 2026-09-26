"""pysrSR is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pysrSR import pysr_regression


def test_pysrSR_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pysr_regression(X=None, y=None)
