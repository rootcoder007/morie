"""kgdsh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgdsh import dk_hermite


def test_kgdsh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk_hermite(data=None)
