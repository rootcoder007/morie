"""agplt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agplt import agplt


def test_agplt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agplt(options=None, setter_ideal=None, reversion=None)
