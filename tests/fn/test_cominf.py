"""cominf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cominf import infomap


def test_cominf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        infomap(G=None)
