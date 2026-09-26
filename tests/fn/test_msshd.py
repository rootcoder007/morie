"""msshd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msshd import shepard_dist


def test_msshd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shepard_dist(data=None)
