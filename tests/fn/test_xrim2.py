"""xrim2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrim2 import sdm_impacts


def test_xrim2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdm_impacts(data=None)
