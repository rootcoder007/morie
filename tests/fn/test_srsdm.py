"""srsdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srsdm import srsdm


def test_srsdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srsdm()
