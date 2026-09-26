"""ppmks is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmks import ppmks


def test_ppmks_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmks()
