"""pssim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pssim import pssim


def test_pssim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pssim()
