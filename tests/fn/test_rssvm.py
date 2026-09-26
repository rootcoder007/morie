"""rssvm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rssvm import rssvm


def test_rssvm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rssvm()
