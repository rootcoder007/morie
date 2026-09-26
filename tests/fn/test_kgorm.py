"""kgorm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgorm import ok_matrix


def test_kgorm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ok_matrix(data=None)
