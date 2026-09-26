"""kgcok is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgcok import cokriging


def test_kgcok_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cokriging(values=None, x=None)
