"""ubnse is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubnse import ubnse


def test_ubnse_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubnse()
