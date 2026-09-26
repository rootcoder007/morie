"""tmkf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmkf import tmkf


def test_tmkf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmkf()
