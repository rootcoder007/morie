"""splgbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.splgbic import splgbic


def test_splgbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        splgbic(ll=None, k=None, n=None)
