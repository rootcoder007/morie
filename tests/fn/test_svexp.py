"""svexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svexp import svexp


def test_svexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svexp()
