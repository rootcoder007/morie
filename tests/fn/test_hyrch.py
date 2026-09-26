"""hyrch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyrch import hyrch


def test_hyrch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyrch()
