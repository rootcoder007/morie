"""aged is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aged import aged


def test_aged_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aged()
