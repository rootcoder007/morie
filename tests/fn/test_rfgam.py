"""rfgam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfgam import rfgam


def test_rfgam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfgam()
