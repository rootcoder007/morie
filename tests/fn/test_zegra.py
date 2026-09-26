"""zegra is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zegra import gravity_access


def test_zegra_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gravity_access(data=None)
