"""kgunr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgunr import uk_residual


def test_kgunr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uk_residual(data=None)
