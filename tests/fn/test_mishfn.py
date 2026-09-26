"""mishfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mishfn import mish_activation


def test_mishfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mish_activation(y=None)
