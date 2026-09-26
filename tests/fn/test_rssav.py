"""rssav is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rssav import rssav


def test_rssav_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rssav()
