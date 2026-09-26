"""dkstf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkstf import dkstf


def test_dkstf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkstf()
