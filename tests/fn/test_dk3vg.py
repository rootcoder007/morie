"""dk3vg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3vg import dk3vg


def test_dk3vg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3vg()
