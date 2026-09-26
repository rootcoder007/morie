"""hsdiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsdiv import hsdiv


def test_hsdiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsdiv()
