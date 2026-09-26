"""cdbias is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdbias import cdbias


def test_cdbias_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdbias()
