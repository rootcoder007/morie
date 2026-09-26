"""aglsm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aglsm import aglsm


def test_aglsm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aglsm()
