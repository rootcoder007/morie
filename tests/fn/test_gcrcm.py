"""gcrcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcrcm import gcrcm


def test_gcrcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcrcm()
