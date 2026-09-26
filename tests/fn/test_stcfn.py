"""stcfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stcfn import stcfn


def test_stcfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stcfn()
