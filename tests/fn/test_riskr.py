"""riskr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.riskr import risch_integration


def test_riskr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        risch_integration(expr=None, x=None)
