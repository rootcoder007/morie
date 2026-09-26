"""aguhi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aguhi import aguhi


def test_aguhi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aguhi()
