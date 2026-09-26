"""ptgfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptgfn import g_function


def test_ptgfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        g_function(data=None)
