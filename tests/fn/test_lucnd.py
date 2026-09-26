"""lucnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lucnd import lucnd


def test_lucnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lucnd()
