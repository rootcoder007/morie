"""ptkfn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptkfn import k_function


def test_ptkfn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        k_function(data=None)
