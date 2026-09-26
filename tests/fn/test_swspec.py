"""swspec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swspec import swspec


def test_swspec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swspec(W=None)
