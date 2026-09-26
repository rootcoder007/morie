"""svbpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svbpl import svbpl


def test_svbpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svbpl()
