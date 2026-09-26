"""ptffn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptffn import f_function


def test_ptffn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        f_function(data=None)
