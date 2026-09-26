"""kgsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgsmp import simple_kriging


def test_kgsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        simple_kriging(values=None, x=None)
