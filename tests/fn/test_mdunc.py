"""mdunc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdunc import mdunc


def test_mdunc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdunc()
