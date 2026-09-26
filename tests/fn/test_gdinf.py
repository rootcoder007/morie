"""gdinf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdinf import gdinf


def test_gdinf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdinf()
