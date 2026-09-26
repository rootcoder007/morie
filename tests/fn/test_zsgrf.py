"""zsgrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsgrf import grid_focal


def test_zsgrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        grid_focal(data=None)
