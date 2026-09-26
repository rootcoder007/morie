"""nmpre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmpre import pre_stat


def test_nmpre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pre_stat(data=None)
