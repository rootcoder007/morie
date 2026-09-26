"""gpwarp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpwarp import gpwarp


def test_gpwarp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpwarp()
