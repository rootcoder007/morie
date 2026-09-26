"""psvot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psvot import psvot


def test_psvot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        psvot()
