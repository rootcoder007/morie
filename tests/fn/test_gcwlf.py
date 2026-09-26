"""gcwlf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcwlf import gcwlf


def test_gcwlf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcwlf()
