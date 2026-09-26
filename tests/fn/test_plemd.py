"""plemd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plemd import plemd


def test_plemd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plemd()
