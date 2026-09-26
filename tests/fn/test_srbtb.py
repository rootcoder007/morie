"""srbtb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srbtb import srbtb


def test_srbtb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srbtb()
