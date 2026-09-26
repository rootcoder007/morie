"""sfmemp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfmemp import sfmemp


def test_sfmemp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfmemp(W=None)
