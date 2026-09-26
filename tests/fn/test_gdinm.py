"""gdinm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdinm import gdinm


def test_gdinm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdinm()
