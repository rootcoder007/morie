"""svqut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svqut import svqut


def test_svqut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svqut()
