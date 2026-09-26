"""sacbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sacbic import sacbic


def test_sacbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sacbic(ll=None, k=None, n=None)
