"""stlin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stlin import stlin


def test_stlin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stlin()
