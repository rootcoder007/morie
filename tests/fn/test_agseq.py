"""agseq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agseq import agseq


def test_agseq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agseq(options=None, setter_ideal=None, reversion=None)
