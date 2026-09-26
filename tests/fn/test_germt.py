"""germt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.germt import germt


def test_germt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        germt()
