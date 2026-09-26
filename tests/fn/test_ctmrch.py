"""ctmrch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctmrch import ctmrch


def test_ctmrch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctmrch()
