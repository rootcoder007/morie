"""svdot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdot import svdot


def test_svdot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svdot()
