"""dtgmb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtgmb import dtgmb


def test_dtgmb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtgmb()
