"""ptjfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptjfn import j_function


def test_ptjfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        j_function(data=None)
