"""spprbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spprbic import spprbic


def test_spprbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spprbic(ll=None, k=None, n=None)
