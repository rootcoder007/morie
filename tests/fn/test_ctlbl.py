"""ctlbl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctlbl import ctlbl


def test_ctlbl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctlbl()
