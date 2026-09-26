"""mtodm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtodm import mtodm


def test_mtodm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtodm()
