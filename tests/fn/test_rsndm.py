"""rsndm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsndm import rsndm


def test_rsndm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsndm()
