"""abrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abrf import abrf


def test_abrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abrf()
