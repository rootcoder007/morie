"""rsseg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsseg import rsseg


def test_rsseg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsseg()
