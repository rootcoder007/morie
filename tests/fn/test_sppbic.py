"""sppbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sppbic import sppbic


def test_sppbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sppbic(ll=None, k=None, n=None)
