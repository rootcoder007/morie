"""bssim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bssim import bssim


def test_bssim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bssim()
