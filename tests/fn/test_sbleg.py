"""sbleg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbleg import sbleg


def test_sbleg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbleg()
