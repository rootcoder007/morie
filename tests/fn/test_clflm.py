"""clflm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clflm import clflm


def test_clflm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clflm()
