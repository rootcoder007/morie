"""filsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.filsm import filsm


def test_filsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        filsm()
