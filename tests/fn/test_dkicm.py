"""dkicm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkicm import dkicm


def test_dkicm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkicm()
