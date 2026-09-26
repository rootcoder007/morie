"""rssrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rssrf import rssrf


def test_rssrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rssrf()
