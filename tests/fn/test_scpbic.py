"""scpbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.scpbic import scpbic


def test_scpbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        scpbic(ll=None, k=None, n=None)
