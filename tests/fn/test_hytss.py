"""hytss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hytss import hytss


def test_hytss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hytss()
