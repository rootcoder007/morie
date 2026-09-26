"""srhhm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srhhm import srhhm


def test_srhhm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srhhm()
