"""nutsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nutsm import nutsm


def test_nutsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nutsm()
