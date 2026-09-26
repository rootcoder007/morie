"""smallw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.smallw import small_world_sigma


def test_smallw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        small_world_sigma(y=None, A=None)
