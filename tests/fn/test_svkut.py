"""svkut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svkut import svkut


def test_svkut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svkut()
