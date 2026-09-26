"""llgaft is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.llgaft import log_logistic_aft


def test_llgaft_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        log_logistic_aft(time=None, event=None, X=None)
