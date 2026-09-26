"""dk3bl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3bl import dk3bl


def test_dk3bl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3bl()
