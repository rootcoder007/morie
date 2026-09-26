"""plwgn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plwgn import plwgn


def test_plwgn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plwgn()
