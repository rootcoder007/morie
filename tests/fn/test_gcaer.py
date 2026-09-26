"""gcaer is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcaer import gcaer


def test_gcaer_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcaer()
