"""mtinf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtinf import mtinf


def test_mtinf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtinf()
