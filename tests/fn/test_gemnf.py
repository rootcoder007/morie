"""gemnf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gemnf import gemnf


def test_gemnf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gemnf()
