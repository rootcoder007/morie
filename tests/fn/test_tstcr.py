"""tstcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tstcr import tstcr


def test_tstcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tstcr()
